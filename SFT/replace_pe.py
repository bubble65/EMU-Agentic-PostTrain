#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace the legacy Chinese system / user prompts in raw rollout JSONL with
the new English prompts in PE.py, while keeping each sample's actual user
question intact.

Layout assumed in input messages:
  messages[0] = {"role": "system",  "content": <legacy SYSTEM_PROMPT_ZH>}
  messages[1] = {"role": "user",    "content": "<legacy USER_PROMPT_ZH>...User: {question}"}
  messages[2:]                      ← assistant/user/tool turns, untouched

Output:
  messages[0]["content"] = SYSTEM_PROMPT
  messages[1]["content"] = USER_PROMPT + question
  (everything else unchanged)

The user question is taken from the top-level `question` field when present;
otherwise it's parsed out of the legacy user content as the substring after
the last "User:" marker.
"""

import argparse
import json
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PE import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402


def _text_of(content: Any) -> str:
    """Extract concatenated text from a message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _set_text(content: Any, new_text: str) -> Any:
    """Replace the text portion of content with new_text while preserving
    non-text blocks (image_url, etc.). String → string; list → list with a
    single new text block followed by the original non-text blocks.
    """
    if isinstance(content, list):
        non_text = [
            b for b in content if isinstance(b, dict) and b.get("type") != "text"
        ]
        return [{"type": "text", "text": new_text}] + non_text
    return new_text


def _extract_question(example: dict, legacy_user_text: str) -> str:
    """Prefer the top-level `question` field; otherwise pull the substring
    after the last 'User:' in the legacy user message."""
    q = (example.get("question") or "").strip()
    if q:
        return q
    idx = legacy_user_text.rfind("User:")
    if idx >= 0:
        return legacy_user_text[idx + len("User:"):].strip()
    return legacy_user_text.strip()


def process_example(example: dict) -> dict:
    msgs: List[dict] = example.get("messages", [])
    if len(msgs) < 2 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        raise ValueError("expected messages[0]=system, messages[1]=user")

    legacy_user_text = _text_of(msgs[1].get("content", ""))
    question = _extract_question(example, legacy_user_text)

    msgs[0]["content"] = _set_text(msgs[0].get("content", ""), SYSTEM_PROMPT)
    msgs[1]["content"] = _set_text(msgs[1].get("content", ""), USER_PROMPT + question)
    return example


def main():
    parser = argparse.ArgumentParser(
        description="Replace legacy Chinese SP/UP with English versions, keeping user question."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most this many records (0 = all)."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_in = n_out = n_skip = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            if args.limit and n_in >= args.limit:
                break
            n_in += 1
            try:
                example = json.loads(line)
                example = process_example(example)
            except Exception as exc:
                print(f"[skip] line {line_no}: {exc}", file=sys.stderr)
                n_skip += 1
                continue
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_out += 1

    print("=" * 50)
    print(f"  Input records:  {n_in}")
    print(f"  Output records: {n_out}")
    print(f"  Skipped:        {n_skip}")
    print("=" * 50)


if __name__ == "__main__":
    main()
