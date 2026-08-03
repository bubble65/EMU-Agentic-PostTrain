import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests


ARK_MODEL_DEFAULT = os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215")
ARK_API_KEY_DEFAULT = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

DEFAULT_MAX_DOC_CHARS = 160000
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REASONING_EFFORT = "medium"

PROMPT_TEMPLATE = """You are an information extraction assistant. Below is a [query] and a piece of [document content].
Please find the most relevant core information from the document related to the query, condense and organize it, with the following requirements:

1. Only keep facts/data/opinions that are truly relevant to the query; remove all unrelated content.
2. Present the information in compact bullet points or short paragraphs. Do not use pleasantries, do not restate the query, and do not include filler phrases like "according to the document".
3. If the document contains no relevant information at all, output only this single line: `NO_RELEVANT_INFO`.
4. Try to preserve specific information from the original text such as key numbers, names, times, sources, etc. Avoid being overly vague.
5. Answer in the same language as the query (if the query is in Chinese, answer in Chinese; if in English, answer in English).

[query]
{query}

[document content]
{doc}

Please directly output the condensed and organized information:
"""

def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("type") == "message":
                    parts.append(_text_from_value(item.get("content")))
        return "".join(parts)
    if isinstance(value, dict):
        if value.get("text"):
            return str(value["text"])
        if value.get("content") is not None:
            return _text_from_value(value.get("content"))
    return ""


def _extract_text_from_resp(data: dict) -> str:
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        text = _text_from_value(msg.get("content"))
        if text:
            return text
        reasoning = msg.get("reasoning_content") or ""
        if reasoning:
            return reasoning

    text = _text_from_value(data.get("content"))
    if text:
        return text

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = data.get("output") or []
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts.append(_text_from_value(item.get("content")))
        text = "".join(parts)
        if text:
            return text

    return ""


class DocReader:
    def __init__(
        self,
        model: str = ARK_MODEL_DEFAULT,
        api_key: str = ARK_API_KEY_DEFAULT,
        base_url: Optional[str] = None,
        max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
        timeout: int = DEFAULT_TIMEOUT,
        max_workers: int = DEFAULT_MAX_WORKERS,
        prompt_template: str = PROMPT_TEMPLATE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        if model_id is not None:
            model = str(model_id)
        if token is not None:
            api_key = token

        self.model = str(model)
        self.api_key = api_key
        self.base_url = base_url or ARK_BASE_URL_DEFAULT
        self.max_doc_chars = int(max_doc_chars)
        self.timeout = int(timeout)
        self.max_workers = int(max_workers)
        self.prompt_template = prompt_template
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.reasoning_effort = str(reasoning_effort)

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _call_ark(self, text: str) -> str:
        if not self.api_key:
            raise RuntimeError("missing ARK_API_KEY; set it in scrpits/run.sh")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        resp = requests.post(
            self.base_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _extract_text_from_resp(resp.json())

    def read(self, query: str, doc: str) -> str:
        if not query or not query.strip():
            return "[reader_error] empty query"
        if not doc or not doc.strip():
            return "NO_RELEVANT_INFO"

        truncated = False
        if len(doc) > self.max_doc_chars:
            doc = doc[: self.max_doc_chars] + "\n...[truncated]"
            truncated = True

        prompt = self.prompt_template.format(query=query.strip(), doc=doc.strip())

        try:
            out = self._call_ark(prompt)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            body = getattr(e.response, "text", "")
            if "ModelNotOpen" in body or "not activated the model" in body:
                return (
                    f"[reader_error] model not open: {self.model}. "
                    "Activate it in Ark Console or set ARK_MODEL to an enabled Doubao model."
                )
            return f"[reader_error] http {status}: {body[:200]}"
        except Exception as e:
            return f"[reader_error] {type(e).__name__}: {e}"

        if not out:
            return "NO_RELEVANT_INFO"

        if truncated and "NO_RELEVANT_INFO" not in out:
            out += (
                "\n\n[note] The original document was too long and has been truncated. "
                "If more complete information is needed, narrow down the query and search again."
            )
        return out

    __call__ = read

    def read_batch(
        self,
        query: str,
        docs: List[str],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        if not docs:
            return []
        workers = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda d: self.read(query, d), docs))


_default_reader: Optional[DocReader] = None


def _get_default_reader() -> DocReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = DocReader()
    return _default_reader


def read_doc(query: str, doc: str) -> str:
    return _get_default_reader().read(query, doc)


if __name__ == "__main__":
    demo_doc = """
    熊二是中国动画片《熊出没》中的主要角色之一,是一只憨厚可爱的棕熊。
    他是熊大的弟弟,平时虽然贪吃懒惰、智商不太高,但是心地善良、力大无穷,
    经常和哥哥熊大一起对抗滥砍滥伐的光头强,保护森林。该动画由华强方特出品,
    自2012年首播以来在中国大陆广受欢迎,衍生出多部大电影。
    与剧情无关的一段:今天天气真好,我中午吃了一份番茄炒蛋盖饭,
    饭店在公司楼下,排队的人很多。
    """
    reader = DocReader()
    print(f"使用模型: {reader.model}")
    print(reader.read("熊二是谁", demo_doc))
