"""Gen-Searcher-style dual reward, adapted to Emu's own outputs + Doubao judge.

Two rewards, both scored on **Emu's own outputs** (no GT image required):

  R_caption  — Gen-Searcher's 5-grade text reward (score ∈ {0, 0.25, 0.5, 0.75, 1.0}).
               Judge is asked: "given the task prompt + the agent's caption,
               would a perfect generator produce a satisfying image?"
  R_image    — Gen-Searcher's 4-dimension world-knowledge image reward.
               Each of {faithfulness, visual_correctness, text_accuracy,
               aesthetics} scored ∈ {0, 0.5, 1}; combined as
                   overall_img = 0.1 f + 0.4 v + 0.4 t + 0.1 a.
               When the prompt does not require readable text, text_accuracy
               defaults to 0.5 (per Gen-Searcher's rubric).

Both rubrics are adapted from the Gen-Searcher reward prompts, with the
GT-image references stripped (we don't have GTs in gen_rl.jsonl) and the
wording flipped from "compare to Image 2" to "compare to what the prompt +
your world knowledge would expect".

The judge endpoint uses the same Doubao / Ark OpenAI-compatible responses
API as ``DataRoller/react_agent.py``. The default judge is the configured
Ark Doubao model, which supports both text and vision through the same
responses surface.

Composition:

    overall = judge_weight   * R_judge
            + format_weight  * R_format
            + draw_weight    * R_draw
            + length_penalty * length_pen
            + no_draw_pen

    R_judge = (1 - alpha) * R_image + alpha * R_caption

Defaults: judge_weight=0.8, format=0.1, draw=0.05, length=0.05, alpha=0.5.
``no_draw_pen = -0.3`` for trajectories that never emitted a native draw —
without it GRPO can't distinguish "no draw" (reward ≈ 0) from "bad draw"
(reward ≈ 0.03) and the policy can collapse toward never drawing.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Optional

REWARD_NAME = "emu_agentic_gensearcher_dual"
REWARD_TYPE = "batch"


# ═════════════════════════════════════════════════════════════════════════
#  Doubao judge configuration — mirrors DataRoller/react_agent.py.
# ═════════════════════════════════════════════════════════════════════════
DOUBAO_MODEL_DEFAULT = os.environ.get(
    "EMU_AGENTIC_JUDGE_MODEL",
    os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
)
DOUBAO_API_KEY_DEFAULT = os.environ.get(
    "EMU_AGENTIC_JUDGE_API_KEY",
    os.getenv("ARK_API_KEY", ""),
)
DOUBAO_BASE_URL_DEFAULT = os.environ.get(
    "EMU_AGENTIC_JUDGE_BASE_URL",
    os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
)
DOUBAO_MAX_OUTPUT_TOKENS_DEFAULT = int(
    os.environ.get("EMU_AGENTIC_JUDGE_MAX_OUTPUT_TOKENS", "1024")
)
DOUBAO_TIMEOUT_DEFAULT = int(os.environ.get("EMU_AGENTIC_JUDGE_TIMEOUT", "120"))

ENABLE_IMAGE_REWARD = str(os.environ.get("EMU_AGENTIC_ENABLE_IMAGE_REWARD", "1")).lower() in {"1", "true", "yes"}
ENABLE_TEXT_REWARD = str(os.environ.get("EMU_AGENTIC_ENABLE_TEXT_REWARD", "1")).lower() in {"1", "true", "yes"}


@lru_cache(maxsize=8)
def _get_doubao_client(model: str, base_url: str, api_key: str) -> Optional[Any]:
    if not api_key:
        return None
    _ = model  # keep model in the cache key for future flexibility
    # This directory contains reward_function/math.py. Temporarily remove it
    # so OpenAI/Pydantic imports the stdlib math module, not the reward file.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    removed: list[str] = []
    for entry in list(sys.path):
        try:
            if os.path.abspath(entry or os.getcwd()) == this_dir:
                sys.path.remove(entry)
                removed.append(entry)
        except OSError:
            continue
    try:
        from openai import OpenAI
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)
    return OpenAI(base_url=base_url, api_key=api_key)


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("output_text"):
                    parts.append(str(item["output_text"]))
                elif item.get("content") is not None:
                    parts.append(_text_from_value(item.get("content")))
        return "".join(parts)
    if isinstance(value, dict):
        if value.get("text"):
            return str(value["text"])
        if value.get("output_text"):
            return str(value["output_text"])
        if value.get("content") is not None:
            return _text_from_value(value.get("content"))
    return ""


def _extract_text_from_resp(data: dict) -> str:
    blocks = data.get("content")
    if isinstance(blocks, list):
        text = _text_from_value(blocks)
        if text:
            return text
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message", {}) or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        if text:
            return text
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = data.get("output") or []
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                parts.append(_text_from_value(item.get("content")))
            elif item.get("type") in {"output_text", "text"}:
                parts.append(_text_from_value(item))
        text = "".join(parts)
        if text:
            return text
    return ""


def _guess_image_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


def _load_image_as_data_url(path: str) -> Optional[str]:
    if not path:
        return None
    try:
        if path.startswith("data:"):
            return path
        if path.startswith(("http://", "https://")):
            return path
        with open(path, "rb") as fh:
            raw = fh.read()
        media_type = _guess_image_media_type(path)
        if not media_type.startswith("image/"):
            media_type = "image/png"
        return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception as exc:
        print(f"[doubao_judge] failed to load image {path}: {exc}")
        return None


def _build_doubao_input(prompt_text: str, image_paths: Optional[list[str]] = None) -> Optional[list[dict[str, Any]]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
    if image_paths:
        n_added = 0
        for path in image_paths:
            data_url = _load_image_as_data_url(path)
            if not data_url:
                continue
            content.append({
                "type": "input_image",
                "image_url": data_url,
                "detail": "auto",
            })
            n_added += 1
        if n_added == 0:
            return None
    return [{"role": "user", "content": content, "partial": False}]


def _call_doubao(prompt_text: str,
                 image_paths: Optional[list[str]] = None,
                 *,
                 model: str = DOUBAO_MODEL_DEFAULT,
                 api_key: str = DOUBAO_API_KEY_DEFAULT,
                 base_url: str = DOUBAO_BASE_URL_DEFAULT,
                 max_output_tokens: int = DOUBAO_MAX_OUTPUT_TOKENS_DEFAULT,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_retries: int = 3,
                 timeout: int = DOUBAO_TIMEOUT_DEFAULT) -> Optional[str]:
    client = _get_doubao_client(model, base_url, api_key)
    if client is None:
        print("[doubao_judge] missing ARK_API_KEY; judge call skipped")
        return None

    input_messages = _build_doubao_input(prompt_text, image_paths=image_paths)
    if input_messages is None:
        print("[doubao_judge] no usable images for judge call")
        return None

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=input_messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                extra_body={"thinking": {"type": "disabled"}},
            )
            resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else {}
            text = _extract_text_from_resp(resp_dict) or getattr(resp, "output_text", "") or ""
            if text:
                return text
            last_err = f"empty content (attempt {attempt + 1})"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    print(f"[doubao_judge] judge call failed: {last_err}")
    return None


# ═════════════════════════════════════════════════════════════════════════
#  Trace parsing — same regexes as the bring-up reward, plus a few more.
# ═════════════════════════════════════════════════════════════════════════
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_TOOL_RESP_RE = re.compile(r"<tool_response>[\s\S]*?</tool_response>")
_BOI_BLOCK_RE = re.compile(r"<\|image start\|>[\s\S]*?<\|image end\|>")
_BOC_RE = re.compile(r"<\|extra_50\|>\s*([\s\S]*?)\s*<\|extra_51\|>")
_THINK_RE = re.compile(r"(<think>[\s\S]*?</think>|<\|extra_60\|>[\s\S]*?<\|extra_61\|>)")
_BSS_RE = re.compile(r"<\|extra_100\|>")
_ESS_RE = re.compile(r"<\|extra_101\|>")


def _extract_grounded_prompt(text: str) -> str:
    matches = _BOC_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def _strip_tool_response_blocks(text: str) -> str:
    return _TOOL_RESP_RE.sub("", text or "")


def _score_format(text: str) -> tuple[float, dict[str, float]]:
    diag: dict[str, float] = {}
    policy_text = _strip_tool_response_blocks(text)
    diag["has_think"] = 1.0 if _THINK_RE.search(text) else 0.0
    tcs = _TOOL_CALL_RE.findall(text)
    diag["n_tool_calls"] = float(len(tcs))
    if tcs:
        good = 0
        for tc in tcs:
            try:
                payload = json.loads(tc.strip())
                if (isinstance(payload, dict) and "name" in payload
                        and "arguments" in payload
                        and isinstance(payload["arguments"], dict)):
                    good += 1
            except Exception:
                pass
        diag["tool_call_validity"] = good / max(1, len(tcs))
    else:
        diag["tool_call_validity"] = 0.0
    diag["has_draw_caption"] = 1.0 if _BOC_RE.search(policy_text) else 0.0
    diag["has_native_draw"] = 1.0 if _BOI_BLOCK_RE.search(policy_text) else 0.0
    n_bss = len(_BSS_RE.findall(text))
    n_ess = len(_ESS_RE.findall(text))
    if max(n_bss, n_ess) == 0:
        diag["spans_balanced"] = 0.0
    else:
        diag["spans_balanced"] = 1.0 - abs(n_bss - n_ess) / max(n_bss, n_ess)
    composite = (
        0.2 * diag["has_think"]
        + 0.25 * diag["tool_call_validity"]
        + 0.15 * diag["has_draw_caption"]
        + 0.25 * diag["has_native_draw"]
        + 0.15 * diag["spans_balanced"]
    )
    return max(0.0, min(1.0, composite)), diag


def _score_draw_signal(text: str) -> float:
    policy_text = _strip_tool_response_blocks(text)
    if _BOI_BLOCK_RE.search(policy_text):
        return 1.0
    if _BOC_RE.search(policy_text):
        return 0.5
    return 0.0


def _length_penalty(n_tokens: int, soft_budget: int = 6000) -> float:
    if n_tokens <= soft_budget:
        return 0.0
    overflow = n_tokens - soft_budget
    return -0.05 * (overflow / 1024.0)


# ═════════════════════════════════════════════════════════════════════════
#  R_caption — Gen-Searcher text reward, no-GT-image variant.
#  Same 5-grade scale {0, 0.25, 0.5, 0.75, 1.0}; same "would a perfect
#  generator produce something matching the user request" framing.
# ═════════════════════════════════════════════════════════════════════════
TEXT_REWARD_VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

_CAPTION_REWARD_SYSTEM = r"""You are an expert evaluator for a text-based image generation pipeline.

You will receive:
1) Task prompt: the original user requirement (what image we want to generate).
2) Model's caption: the natural-language caption the agent produced just before
   it asked its native image-generator to render the picture.

Your task (TEXT-only — there is NO ground-truth image):
- Judge how well this caption would support generating an image that matches
  the user requirement.
- You are NOT evaluating an actual generated image here. You are evaluating
  whether the caption is well-aligned with the task: if we had a perfect
  image generator, would this caption be sufficient to satisfy the user?
- Consider: Does the caption capture the key requirements from the task
  (subjects/identities, setting, props, relations/counts, required style,
  any readable text or grounded fact the task names)? Are there critical
  missing or wrong elements?

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys (rationale first,
then score):
{
  "rationale": string,
  "score": number
}

Rationale requirements (MANDATORY):
- Start with: "Constraints:" and list the extracted hard constraints (2-5,
  or more if needed) from the task prompt (required subjects/identities,
  setting, style, key props, readable text if any, etc.).
- After listing constraints, in 2-6 more sentences give evidence-based
  rationale: cite the task and the caption; state why the score is justified.
- If you cannot identify the constraints, you must still list what you
  believe are the hard constraints.
- Total rationale: 5-10 short sentences.

SCORING SCALE (VERY IMPORTANT):
- "score" MUST be exactly one of: 0, 0.25, 0.5, 0.75, 1.0

1.0 (Exemplary): The caption is fully sufficient. It perfectly aligns with
the task; a perfect generator would produce an image that satisfies every
hard constraint.

0.75 (Very good): Strong alignment; at most minor gaps or imprecisions.

0.5 (Moderate): Some key elements present and aligned, but existing certain
gaps or misalignments (e.g., missing a key subject, vague on a key
distinctive feature, part of required text is not correct).

0.25 (Weak): Significant missing or wrong elements; the caption would
likely produce a clearly different or incomplete image.

0 (Poor): The caption does not support generating an image that satisfies
the task (wrong focus, missing critical requirements, off-topic, gibberish,
or empty).

Output JSON only. No markdown. No extra text."""


def _build_caption_judge_prompt(question: str, caption: str) -> str:
    return _CAPTION_REWARD_SYSTEM + "\n\n" + (
        f"Task prompt (what image we want to generate):\n{question}\n\n"
        f"Model's caption (the agent's final prompt fed to its image generator):\n{caption}\n\n"
        "Output JSON with 'rationale' and 'score' (one of 0, 0.25, 0.5, 0.75, 1.0)."
    )


def _parse_caption_score(raw: str) -> Optional[float]:
    """Snap to nearest valid score in TEXT_REWARD_VALID_SCORES, or None."""
    if not raw:
        return None
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    start = txt.find("{")
    end = txt.rfind("}")
    snippet = txt[start:end + 1] if (start >= 0 and end > start) else txt
    s_val: Optional[float] = None
    try:
        obj = json.loads(snippet)
        v = obj.get("score")
        if isinstance(v, (int, float)):
            s_val = float(v)
    except Exception:
        pass
    if s_val is None:
        m = re.search(r'"score"\s*:\s*([0-9.]+)', txt)
        if m:
            try:
                s_val = float(m.group(1))
            except Exception:
                pass
    if s_val is None:
        return None
    best = TEXT_REWARD_VALID_SCORES[0]
    for v in TEXT_REWARD_VALID_SCORES:
        if abs(s_val - v) < abs(s_val - best):
            best = v
    return max(0.0, min(1.0, float(best)))


def _caption_judge(question: str,
                   caption: str,
                   *,
                   judge_model: str,
                   judge_api_key: str,
                   judge_base_url: str) -> tuple[float, dict[str, Any]]:
    if not caption:
        return 0.0, {"caption_status": "no_caption"}
    prompt = _build_caption_judge_prompt(question.strip(), caption.strip())
    raw = _call_doubao(
        prompt,
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        max_output_tokens=512,
        temperature=0.0,
        max_retries=2,
    )
    if raw is None:
        # Network failure heuristic — keep gradient alive: short caption → low
        # score, long caption → middling score. Better than collapsing to 0.
        heuristic = min(len(caption) / 600.0, 1.0) * 0.5
        return float(heuristic), {"caption_status": "fallback_heuristic",
                                  "caption_raw": float(heuristic)}
    score = _parse_caption_score(raw)
    if score is None:
        return 0.0, {"caption_status": "parse_fail", "caption_raw_text": raw[:200]}
    return float(score), {"caption_status": "ok", "caption_raw": float(score)}


# ═════════════════════════════════════════════════════════════════════════
#  R_image — Gen-Searcher worldgen reward, no-GT-image variant.
#  Same 4 dimensions (faithfulness / visual_correctness / text_accuracy /
#  aesthetics) on a 3-grade {0, 0.5, 1} scale, combined as
#      0.1*f + 0.4*v + 0.4*t + 0.1*a.
#  "visual_correctness" and "text_accuracy" no longer compare to a GT image
#  (we have none); instead they ask whether the prompt-required, world-
#  knowledge-checkable details are correctly realized in the generated image.
# ═════════════════════════════════════════════════════════════════════════
_IMAGE_REWARD_SYSTEM = r"""You are a strict and professional expert evaluator
for AI-generated images grounded with world knowledge (MODEL EVALUATION).

You will receive:
1) A task prompt (what the image must show).
2) The generated image (model output to be evaluated).

There is NO ground-truth reference image. Use your own world knowledge as the
implicit reference: for any externally-checkable detail named in the prompt
(named person/landmark/badge/logo/year/title/etc.), judge whether the image
clearly and correctly realizes it.

All the input images are AI-generated. All humans in the images are AI-
generated too, so you need not worry about the privacy confidentials.

Critical clarification (VERY IMPORTANT):
- This is NOT a pixel-level similarity task.
- Focus on whether prompt-required, externally-checkable (world-knowledge)
  details are correctly AND verifiably realized in the image.
- Do NOT assume correctness if a key detail is not clearly visible/readable.
  If unverifiable, score lower.

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys:
{
  "rationale": string,
  "faithfulness": number,
  "visual_correctness": number,
  "text_accuracy": number,
  "aesthetics": number,
  "text_accuracy_na": boolean
}
SCORING SCALE (VERY IMPORTANT):
- Each of faithfulness / visual_correctness / text_accuracy / aesthetics
  MUST be exactly one of: 0, 0.5, 1
- 1 (Exemplary) is rare and requires perfect success for that dimension.
- 0.5 (Conditional) means mostly correct but not perfect.
- 0 (Rejected) means failed on important requirements.
- "rationale" must be 5-10 short sentences, evidence-based, referring only
  to what is visible.
- "text_accuracy_na" should be true if the prompt does not require any
  readable text, otherwise false. When true, set "text_accuracy": 0.5.

Implicit required step (ENFORCED via rationale):
- In the rationale, you MUST explicitly list the extracted prompt hard
  constraints (2-5, or more if needed) BEFORE scoring. If you cannot
  identify the constraints, you must still list what you believe are the
  hard constraints.

==========================
STRICT 3-LEVEL RUBRICS
(Each dimension uses ONLY {0, 0.5, 1})
==========================

1) faithfulness (overall prompt adherence: presence & structure only):
- Whether the image includes the prompt-requested elements and scene
  structure (who/what is present, what is happening, where it happens, and
  the required style/format).
(Exemplary) Score = 1 ONLY IF:
- The image clearly includes everything the prompt asks for in terms of
  visible content and structure: all required subjects/entities are
  present, the required setting and key props appear, required
  actions/relations/counts are shown, and the required style/format is
  followed.
- Any required in-scene evidence elements requested by the prompt (e.g., a
  plaque/sign, a map, a report paper, a badge) are present as elements.
(Conditional) Score = 0.5 ONLY IF:
- Includes almost all prompt-requested content with only minor omissions
  or minor staging differences that do not change what the scene depicts.
(Rejected) Score = 0 IF:
- One or more prompt-requested essential elements are not shown at all, or
  the scene structure clearly does not match the prompt's request.

2) visual_correctness (world-knowledge-grounded visual identity is the
core; extremely strict):
(Exemplary) Score = 1 ONLY IF:
- The prompt-required primary subjects/objects in the image match what
  world knowledge expects in their stable visual features with NO
  substantive mismatches: same face/hairstyle silhouette one would expect
  for the named person, same emblem/logo geometry for a named brand, same
  landmark facade for a named building, same armor/clothing design and
  key colors/patterns for a named character, same distinctive props/object
  geometry, etc.
(Conditional) Score = 0.5 ONLY IF:
- The image can still be considered the correct visual instance, with
  minor variations in face/armor/colors/props but the overall identity
  remains recognizable and broadly consistent with what world knowledge
  expects.
- IMPORTANT: "same role archetype" (generic knight/princess/warrior) alone
  does NOT qualify for 0.5 when the prompt names a specific identity.
(Rejected) Score = 0 IF:
- Any substantive mismatch vs world-knowledge expectations on stable visual
  features (different face/hair/armor design/color scheme/emblem/prop
  geometry/landmark cues), even if the overall scene still looks plausible.

3) text_accuracy (required readable text; ALL relevant text must be
correct AND very clearly readable; NO partial credit for wrong text):
Rule:
- If the prompt does NOT require any readable text: output
  "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON. State in
  the rationale that the prompt did not require readable text.
- If the prompt DOES require readable text: output "text_accuracy_na":
  false and score per the criteria below.
(Exemplary) Score = 1 ONLY IF:
- ALL required text AND any prompt-involved text elements are:
  (a) present, (b) very clearly readable (crisp, unambiguous), (c) correct
  and consistent with the prompt's requirements.
(Conditional) Score = 0.5 ONLY IF:
- Much of the required/prompt-involved text is readable and generally
  correct; although parts may contain inaccuracies or omissions, the
  overall meaning remains clear and not seriously inconsistent with the
  prompt requirements.
(Rejected) Score = 0 IF:
- Any required/prompt-involved text is missing, unclear, not very readable,
  gibberish, placeholder, OR incorrect.
- Even if perfectly readable, if content is not correct, text_accuracy
  MUST be 0.

4) aesthetics:
(Exemplary) Score = 1 ONLY IF:
- Masterpiece-level composition and polish.
(Conditional) Score = 0.5 ONLY IF:
- Very beautiful and polished, OR slightly less refined than top-tier.
(Rejected) Score = 0 IF:
- Merely average/OK-looking, OR cluttered/awkward framing, OR visible
  artifacts/noise that harm the overall appeal.

Rationale requirements (MANDATORY):
- Start with: "Constraints:" and list the extracted constraints (2-5, or
  more if needed).
- State whether the prompt required readable text; if not required, output
  "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON and say so
  in the rationale.
- Mention 2-5 key checks against world-knowledge expectations (NOT
  demanding identical layout).
- Keep within 10 sentences.

Output JSON only. No markdown. No extra text."""


def _build_image_judge_prompt(question: str, caption: str) -> str:
    return _IMAGE_REWARD_SYSTEM + "\n\n" + (
        f"Task prompt (what the image must show):\n{question}\n\n"
        f"Agent's caption used to drive the generator (for context only):\n{caption}\n\n"
        "Output a single JSON object with the four scores, text_accuracy_na,"
        " and the short rationale."
    )


def _parse_image_scores(raw: str) -> Optional[dict[str, Any]]:
    """Return {f, v, t, a, text_na, rationale} normalized to {0,0.5,1}, or None."""
    if not raw:
        return None
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    start = txt.find("{")
    end = txt.rfind("}")
    snippet = txt[start:end + 1] if (start >= 0 and end > start) else txt
    obj: Any
    try:
        obj = json.loads(snippet)
    except Exception:
        snippet = re.sub(r",\s*}", "}", snippet)
        snippet = re.sub(r",\s*]", "]", snippet)
        try:
            obj = json.loads(snippet)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None

    def snap_3(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        best = 0.0
        for cand in (0.0, 0.5, 1.0):
            if abs(v - cand) < abs(v - best):
                best = cand
        return float(max(0.0, min(1.0, best)))

    f = snap_3(obj.get("faithfulness", 0))
    v = snap_3(obj.get("visual_correctness", 0))
    a = snap_3(obj.get("aesthetics", 0))
    text_na = obj.get("text_accuracy_na")
    is_na = text_na in (True, "true", "True", 1)
    t = 0.5 if is_na else snap_3(obj.get("text_accuracy", 0))
    rationale = str(obj.get("rationale", ""))[:500]
    return {"f": f, "v": v, "t": t, "a": a, "text_na": bool(is_na),
            "rationale": rationale}


def _image_judge(question: str,
                 caption: str,
                 image_path: str,
                 *,
                 judge_model: str,
                 judge_api_key: str,
                 judge_base_url: str) -> tuple[float, dict[str, Any]]:
    if not image_path or not os.path.exists(image_path):
        return 0.0, {"image_status": "no_image", "image_path": str(image_path)}
    prompt_text = _build_image_judge_prompt(question.strip(),
                                            (caption or "").strip() or "(empty)")
    raw = _call_doubao(
        prompt_text,
        [image_path],
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        max_output_tokens=800,
        temperature=0.0,
        max_retries=2,
    )
    if raw is None:
        return 0.0, {"image_status": "vision_fail", "image_path": image_path}
    parsed = _parse_image_scores(raw)
    if parsed is None:
        return 0.0, {"image_status": "parse_fail",
                     "image_raw_text": raw[:200],
                     "image_path": image_path}
    overall = round(0.1 * parsed["f"] + 0.4 * parsed["v"]
                    + 0.4 * parsed["t"] + 0.1 * parsed["a"], 3)
    overall = max(0.0, min(1.0, overall))
    return float(overall), {"image_status": "ok",
                            "image_f": parsed["f"],
                            "image_v": parsed["v"],
                            "image_t": parsed["t"],
                            "image_a": parsed["a"],
                            "image_text_na": float(parsed["text_na"]),
                            "image_path": image_path}


# ═════════════════════════════════════════════════════════════════════════
#  Per-sample pipeline + batched compute_score (verl entry point).
# ═════════════════════════════════════════════════════════════════════════
def _process_one(sample: dict[str, Any],
                 *,
                 enable_caption: bool,
                 enable_image: bool,
                 max_caption_chars: int,
                 judge_model: str,
                 judge_api_key: str,
                 judge_base_url: str) -> dict[str, Any]:
    text = sample.get("response", "") or ""
    if not isinstance(text, str):
        text = str(text)
    # Prefer the agentic question (untouched user prompt) over ground_truth
    # (which the dataset echoes from the same source but may have been
    # truncated by max_prompt_length).
    question = sample.get("question") or sample.get("ground_truth") or ""
    if not isinstance(question, str):
        question = str(question or "")

    caption = _extract_grounded_prompt(text)
    if max_caption_chars and len(caption) > max_caption_chars:
        caption = caption[:max_caption_chars]

    policy_text = _strip_tool_response_blocks(text)
    has_native_draw = bool(_BOI_BLOCK_RE.search(policy_text))

    diag: dict[str, Any] = {
        "caption_chars": len(caption),
        "has_native_draw": has_native_draw,
    }

    # R_caption
    if enable_caption:
        r_caption, c_diag = _caption_judge(
            question,
            caption,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
        )
    else:
        r_caption, c_diag = 0.0, {"caption_status": "disabled"}
    diag.update(c_diag)

    # R_image (Emu's own PNG, no GT)
    img_path = sample.get("rollout_image_path") or ""
    if not isinstance(img_path, str):
        img_path = str(img_path)
    if enable_image and img_path and os.path.exists(img_path):
        r_image, i_diag = _image_judge(
            question,
            caption,
            img_path,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
        )
    elif not enable_image:
        r_image, i_diag = 0.0, {"image_status": "disabled"}
    elif not img_path:
        r_image, i_diag = 0.0, {"image_status": "no_rollout_image_path"}
    else:
        r_image, i_diag = 0.0, {"image_status": "missing_file",
                                "image_path": img_path}
    diag.update(i_diag)

    return {"r_caption": float(r_caption), "r_image": float(r_image),
            "diag": diag, "caption": caption}


def compute_score(
    reward_inputs: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    format_weight: float = 0.1,
    draw_weight: float = 0.05,
    length_penalty_weight: float = 0.05,
    soft_length_budget: int = 6000,
    judge_weight: float = 0.8,
    no_image_penalty: float = 0.3,
    max_workers: int = 8,
    max_caption_chars: int = 4000,
    enable_caption_reward: Optional[bool] = None,
    enable_image_reward: Optional[bool] = None,
    # Compat shims so the same yaml works for any reward we might swap in.
    vlm_weight: float = 0.0,
    vlm_api_base: str = "",
    vlm_api_key: str = "",
    vlm_model_name: str = "",
    enable_vlm_reward: bool = False,
    max_vlm_workers: int = 0,
    **_unused: Any,
) -> list[dict[str, float]]:
    """Gen-Searcher-style dual reward on Emu's own caption + image.

    Per-sample:

        R_judge   = (1 - alpha) * R_image + alpha * R_caption
        overall   = judge_weight * R_judge
                  + format_weight * format
                  + draw_weight * draw_signal
                  + length_penalty_weight * length_pen
                  + no_draw_pen
    """
    enable_caption = (enable_caption_reward
                      if enable_caption_reward is not None
                      else ENABLE_TEXT_REWARD)
    enable_image = (enable_image_reward
                    if enable_image_reward is not None
                    else ENABLE_IMAGE_REWARD)
    judge_model = (vlm_model_name or DOUBAO_MODEL_DEFAULT).strip() or DOUBAO_MODEL_DEFAULT
    judge_api_key = vlm_api_key or DOUBAO_API_KEY_DEFAULT
    judge_base_url = vlm_api_base or DOUBAO_BASE_URL_DEFAULT

    n = len(reward_inputs)
    if n == 0:
        return []

    results: list[Optional[dict[str, Any]]] = [None] * n
    workers = max(1, min(int(max_workers), n))
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one, sample,
                      enable_caption=enable_caption,
                      enable_image=enable_image,
                      max_caption_chars=max_caption_chars,
                      judge_model=judge_model,
                      judge_api_key=judge_api_key,
                      judge_base_url=judge_base_url): i
            for i, sample in enumerate(reward_inputs)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f"[gensearcher_dual] worker {i} crashed: {exc!r}")
                results[i] = {"r_caption": 0.0, "r_image": 0.0,
                              "diag": {"worker_error": repr(exc)},
                              "caption": ""}

    out: list[dict[str, float]] = []
    caption_vals: list[float] = []
    image_vals: list[float] = []
    n_has_image_path = 0
    n_image_judged_ok = 0

    for i, sample in enumerate(reward_inputs):
        text = sample.get("response", "") or ""
        if not isinstance(text, str):
            text = str(text)
        n_tokens = int(sample.get("response_length", 0) or 0)

        fmt, fmt_diag = _score_format(text)
        draw_sig = _score_draw_signal(text)
        length_pen = _length_penalty(n_tokens, soft_budget=soft_length_budget)

        res = results[i] or {"r_caption": 0.0, "r_image": 0.0, "diag": {},
                             "caption": ""}
        r_caption = float(res["r_caption"])
        r_image = float(res["r_image"])

        if enable_caption and enable_image:
            r_judge = (1.0 - alpha) * r_image + alpha * r_caption
        elif enable_caption:
            r_judge = r_caption
        elif enable_image:
            r_judge = r_image
        else:
            r_judge = 0.0

        # has_draw: use a TWO-source check. The response string regex misses
        # trajectories where the final <|image start|>...<|image end|> block was
        # truncated by max_response_length (force_draw with long tool_response
        # context regularly produces 16k+ token responses; the visual block sits
        # at the end and gets sliced). The rollout's ``saved_images`` list, on
        # the other hand, is the ground truth — it was set if and only if the
        # VQ decoder actually wrote a PNG for this trajectory. Trust whichever
        # source says "draw happened".
        regex_has_draw = bool(_BOI_BLOCK_RE.search(_strip_tool_response_blocks(text)))
        saved_paths = sample.get("saved_images") or []
        path_has_draw = bool(saved_paths) and bool(
            sample.get("rollout_image_path") or ""
        )
        has_draw = regex_has_draw or path_has_draw
        no_draw_pen = 0.0 if has_draw else -float(no_image_penalty)

        overall = (
            judge_weight * r_judge
            + format_weight * fmt
            + draw_weight * draw_sig
            + length_penalty_weight * length_pen
            + no_draw_pen
        )

        row: dict[str, float] = {
            "overall": float(overall),
            "r_caption": float(r_caption),
            "r_image": float(r_image),
            "r_judge": float(r_judge),
            "no_draw_penalty": float(no_draw_pen),
            "accuracy": float(r_judge),     # verl logger picks up "accuracy"
            "format": float(fmt),
            "draw": float(draw_sig),
            "length_pen": float(length_pen),
        }
        diag = res["diag"]
        for k, v in fmt_diag.items():
            row[f"fmt_{k}"] = float(v)
        for k, v in diag.items():
            if isinstance(v, (int, float, bool)):
                row[f"d_{k}"] = float(v)
        out.append(row)
        caption_vals.append(r_caption)
        image_vals.append(r_image)
        if sample.get("rollout_image_path"):
            n_has_image_path += 1
        if diag.get("image_status") == "ok":
            n_image_judged_ok += 1

    if out:
        n_no_draw = sum(1 for r in out if r.get("no_draw_penalty", 0.0) < 0)
        try:
            print(
                f"[gensearcher_dual] batch n={n} "
                f"R_caption mean={sum(caption_vals)/n:.3f} "
                f"R_image mean={sum(image_vals)/n:.3f} "
                f"R_judge mean={sum((1-alpha)*ji + alpha*jt for ji, jt in zip(image_vals, caption_vals))/n:.3f} "
                f"overall mean={sum(r['overall'] for r in out)/n:.3f} "
                f"native_imgs={n_has_image_path}/{n} judged_ok={n_image_judged_ok}/{n} "
                f"no_draw={n_no_draw}/{n} (penalty=-{no_image_penalty:.2f}) "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
        except Exception:
            pass

    return out


# Smoke test (no network).
if __name__ == "__main__":
    fake_trace = (
        "<|extra_60|>think<|extra_61|>\n"
        "<tool_call>{\"name\": \"text_search\", \"arguments\": {\"query\": [\"x\"]}}</tool_call>\n"
        "<tool_response>...</tool_response><|extra_100|>"
        "<|extra_60|>more think<|extra_61|>"
        "<|extra_50|>a fish in a lab tank<|extra_51|>"
        "<|image start|>1024*1024<|image end|>"
    )
    rows = [{"response": fake_trace, "response_length": 1000,
             "question": "draw a Danionella fish in a lab",
             "ground_truth": "",
             "rollout_image_path": ""}]
    print(compute_score(rows, enable_caption_reward=False,
                        enable_image_reward=False))
