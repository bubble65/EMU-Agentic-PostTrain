"""Reformat search-tool JSON returns into the EXACT in-training tool_response
layout so the model sees its training distribution at inference time.

Training-time format (per convert_v2.py + observed sft.jsonl):

    text_search:
        \n
        query : <q1>\n
        查询结果 : <title> | <abstract>\n
        \n
        query : <q2>\n
        查询结果 : ...\n

    image_search:
        \n
        query : <q1>\n
        查询结果 : <desp>{html-ish reader summary}</desp>[IMAGE N] <BoI>...<EoI>\n
        \n
        query : <q2>\n
        查询结果 : <desp>...</desp>[IMAGE M] <BoI>...<EoI>\n

Failed queries become `查询结果 : [error] <reason>`.
`<desp>...</desp>` wraps the reader summary so the model can tell it apart
from page-text/title.

Public API
----------
    reformat_tool_response(
        tool_name, tool_result_str, server, label_book,
    ) -> str
        Return a string ready to drop inside <tool_response>...</tool_response>.
        For image_search, encodes returned URLs via server.encode_images and
        emits `[IMAGE N] <BoI>...<EoI>` inline after the corresponding
        query's 查询结果 line.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ── Public entry point ───────────────────────────────────────────────────


def reformat_tool_response(
    tool_name: str,
    tool_result_str: str,
    server,           # EmuServerClient (encode_images)
    label_book,       # _ImageLabelBook (assigns global [IMAGE N] labels)
) -> str:
    """Return the body that goes inside <tool_response>...</tool_response>.

    Falls back to returning `tool_result_str` verbatim if parsing fails — we
    never want to crash the agent loop over a formatting issue.
    """
    if not tool_result_str:
        return tool_result_str
    try:
        parsed = _safe_loads(tool_result_str)
    except Exception:
        return tool_result_str
    if parsed is None:
        return tool_result_str

    if tool_name == "text_search":
        return _format_text_search(parsed)
    if tool_name == "image_search":
        return _format_image_search(parsed, server, label_book)
    # Unknown tool — leave it alone.
    return tool_result_str


# ── text_search ──────────────────────────────────────────────────────────


def _format_text_search(parsed: Any) -> str:
    """Per-query block:

        query : <q>
        查询结果 : <title> | <content>

    Joined with a blank line between queries. Leading "\n" matches training.
    """
    items = _as_list_of_dicts(parsed)
    if not items:
        return _empty_response()

    blocks: List[str] = []
    for item in items:
        q = _str_field(item, "query")
        if not item.get("ok", True):
            err = _str_field(item, "error") or "search failed"
            blocks.append(_block(q, f"[error] {err}"))
            continue
        rank1 = item.get("rank1") or {}
        if not isinstance(rank1, dict):
            blocks.append(_block(q, "[error] malformed result"))
            continue
        title = _str_field(rank1, "title")
        content = _str_field(rank1, "content")
        if not content:
            blocks.append(_block(q, "[no content]"))
            continue
        # Match training shape: "<title> | <content>" — title may be empty,
        # in which case skip the leading "| ".
        if title:
            body = f"{title} | {content}"
        else:
            body = content
        blocks.append(_block(q, body))

    return "\n" + "\n\n".join(blocks) + "\n"


# ── image_search ─────────────────────────────────────────────────────────


def _format_image_search(parsed: Any, server, label_book) -> str:
    """Per-query block:

        query : <q>
        查询结果 : <desp>{summary}</desp>[IMAGE N] <BoI>...<EoI>

    If multiple results came back for one query, we emit only the FIRST
    one's image inline (matching the training data — one [IMAGE N] per
    query block). The summary used is that first result's `summary`
    field; if it starts with `[error]` or is empty, we mark the block as
    failed and skip image encoding for it.
    """
    items = _as_list_of_dicts(parsed)
    if not items:
        return _empty_response()

    # First pass: collect (query, summary, url, ok). We batch-encode all
    # URLs in one /encode_images call to amortize the round-trip.
    plan: List[Tuple[str, str, Optional[str], bool, str]] = []
    # tuples: (query, summary, url_or_None, ok, error_msg)
    for item in items:
        q = _str_field(item, "query")
        if not item.get("ok", True):
            err = _str_field(item, "error") or "search failed"
            plan.append((q, "", None, False, err))
            continue
        results = item.get("results") or []
        if not isinstance(results, list) or not results:
            plan.append((q, "", None, False, "No image results"))
            continue
        first = results[0] if isinstance(results[0], dict) else {}
        url = _str_field(first, "url")
        summary = _str_field(first, "summary")
        # If reader said NO_RELEVANT_INFO or [reader_error], training-style
        # still gets the query block but with summary-as-error and no image.
        if summary.startswith("[reader_error]") or summary.strip() == "NO_RELEVANT_INFO":
            plan.append((q, summary, None, False, summary))
            continue
        if not url:
            plan.append((q, summary, None, False, "missing url"))
            continue
        plan.append((q, summary, url, True, ""))

    # Encode the ok urls.
    urls_to_encode = [p[2] for p in plan if p[3] and p[2]]
    enc_map: Dict[str, Tuple[bool, str]] = {}
    if urls_to_encode:
        try:
            enc = server.encode_images(urls_to_encode)
        except Exception as exc:
            # Server side hiccup — fall back gracefully.
            enc = {"tokens": [], "ok": [], "errors": [str(exc)] * len(urls_to_encode)}
        tokens = enc.get("tokens", [])
        oks = enc.get("ok", [])
        for u, t, o in zip(urls_to_encode, tokens, oks):
            enc_map[u] = (bool(o), t or "")

    # Assemble blocks.
    blocks: List[str] = []
    for q, summary, url, ok, err in plan:
        if not ok:
            # Failed block — keep a stub so the model still sees that we tried.
            payload = f"[error] {err}" if err else "[error] no image"
            blocks.append(_block(q, payload))
            continue
        enc_ok, tok = enc_map.get(url, (False, ""))
        label = label_book.assign(url)
        if enc_ok and tok:
            chaxun = f"<desp>{summary}</desp>{label} {tok}"
        else:
            chaxun = f"<desp>{summary}</desp>{label} [image load failed]"
        blocks.append(_block(q, chaxun))

    return "\n" + "\n\n".join(blocks) + "\n"


# ── shared helpers ───────────────────────────────────────────────────────


def _block(query: str, chaxun_body: str) -> str:
    """One query/result block, no leading or trailing newline — caller
    glues blocks together with \\n\\n."""
    q = (query or "").strip() or "(empty query)"
    return f"query : {q}\n查询结果 : {chaxun_body}"


def _empty_response() -> str:
    return "\n[no results]\n"


def _safe_loads(text: str) -> Any:
    """json.loads that returns None on failure rather than raising. We treat
    any non-JSON input as 'leave the body alone' upstream."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _as_list_of_dicts(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _str_field(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


# ── think rewrite (also lives here for one-stop shopping) ────────────────


_THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.IGNORECASE)

# Special tokens — keep in sync with template.SPECIAL.
_BOG = "<|extra_60|>"
_EOG = "<|extra_61|>"


def rewrite_think_to_bog_eog(body: str) -> str:
    """Replace `<think> X </think>` with `<|extra_60|> X <|extra_61|>\\n`
    everywhere in `body`. Trailing newline matches convert_v2 line 574.
    Idempotent: if BoG/EoG are already there, the regex won't fire.
    """
    if not body:
        return body
    if "<think>" not in body and "</think>" not in body:
        return body

    def _sub(m: re.Match) -> str:
        inner = (m.group(1) or "").strip()
        return f"{_BOG} {inner} {_EOG}\n"

    return _THINK_RE.sub(_sub, body)
