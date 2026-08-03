import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from tool_reader import DocReader

SERPER_API_KEY_DEFAULT = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL_DEFAULT = os.getenv("SERPER_BASE_URL", "https://google.serper.dev/search")
DEEP_SEARCH_PROXY = os.getenv("DEEP_SEARCH_PROXY", "")

JINA_BASE_URL_DEFAULT = "https://r.jina.ai/"
JINA_PROXY_DEFAULT = os.getenv("DEEP_BROWSE_PROXY", "")
JINA_BAD_DOMAINS_DEFAULT: List[str] = [
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "weibo.com",
    "douyin.com",
    "youtube.com",
    "youtu.be",
    "pinterest.com",
    "linkedin.com",
    "xiaohongshu.com",
    "redbook.com",
    "zhihu.com/video",
]
MIN_USEFUL_CHARS_DEFAULT = 50


def _proxies(proxy: str) -> Optional[Dict[str, str]]:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _domain_of(url: str) -> str:
    try:
        m = re.match(r"^https?://([^/]+)", url, re.IGNORECASE)
        if not m:
            return ""
        return m.group(1).lower().split(":")[0]
    except Exception:
        return ""


def _is_bad_domain(url: str, bad_domains: List[str]) -> bool:
    host = _domain_of(url)
    if not host:
        return False
    lower_url = url.lower()
    for bad in bad_domains:
        bad = bad.lower().strip()
        if not bad:
            continue
        if "/" in bad:
            if bad in lower_url:
                return True
        elif host == bad or host.endswith("." + bad):
            return True
    return False


def search_google(
    query: str,
    topk: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    proxy: Optional[str] = None,
    timeout: int = 100,
) -> List[Dict[str, Any]]:
    resolved_api_key = api_key or SERPER_API_KEY_DEFAULT
    if not resolved_api_key:
        raise RuntimeError("missing SERPER_API_KEY; set it in scrpits/run.sh")

    resp = requests.post(
        base_url or SERPER_BASE_URL_DEFAULT,
        headers={
            "X-API-KEY": resolved_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": topk},
        proxies=_proxies(proxy) if proxy is not None else None,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("organic", []) or []


@register_tool("text_search", allow_overwrite=True)
class TextSearch(BaseTool):
    name = "text_search"
    description = "Text search with public Jina page fetch."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search queries.",
            },
            "count": {
                "type": "integer",
                "description": "Number of search results to keep.",
                "default": 10,
            },
            "fetch_rank1": {
                "type": "boolean",
                "description": "Whether to fetch and summarize the best result.",
                "default": True,
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        cfg = cfg or {}

        self.api_key: str = cfg.get("serper_api_key", SERPER_API_KEY_DEFAULT)
        self.base_url: str = cfg.get("serper_base_url", SERPER_BASE_URL_DEFAULT)
        self.proxy: str = cfg.get("serper_proxy", DEEP_SEARCH_PROXY)
        self.timeout: int = int(cfg.get("timeout", 100))
        self.max_retries: int = int(cfg.get("max_retries", 3))
        self.max_workers: int = int(cfg.get("max_workers", 4))
        self.rank_fallback_max: int = int(cfg.get("rank_fallback_max", 4))
        self.rank1_max_chars: int = int(cfg.get("rank1_max_chars", 1000))
        self.page_max_chars: int = int(cfg.get("page_max_chars", 128000))
        self.min_useful_chars: int = int(cfg.get("min_useful_chars", MIN_USEFUL_CHARS_DEFAULT))
        self.jina_base_url: str = cfg.get("jina_base_url", JINA_BASE_URL_DEFAULT)
        self.jina_proxy: str = cfg.get("jina_proxy", JINA_PROXY_DEFAULT)
        self.jina_bad_domains: List[str] = list(
            cfg.get("jina_bad_domains", JINA_BAD_DOMAINS_DEFAULT)
        )
        self.reader: DocReader = cfg.get("reader") or DocReader()

    def _error(self, query: str, error: str = "") -> Dict[str, Any]:
        return {
            "ok": False,
            "tool": "text_search",
            "query": query,
            "count": 0,
            "results": [],
            "rank1": None,
            "error": error,
        }

    def _normalize_results(
        self, raw: List[Dict[str, Any]], count: int
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw[:count], 1):
            if item.get("extra_snippets"):
                snippet = "\n".join(item["extra_snippets"])
            else:
                snippet = item.get("snippet") or item.get("description", "")
            results.append(
                {
                    "rank": idx,
                    "title": item.get("title", ""),
                    "url": item.get("link") or item.get("url", ""),
                    "snippet": (snippet or "")[:2000],
                    "domain": item.get("displayLink", ""),
                    "published_date": item.get("date", ""),
                }
            )
        return results

    def _search_google(self, query: str, count: int) -> List[Dict[str, Any]]:
        try:
            raw = search_google(
                query,
                topk=count,
                api_key=self.api_key,
                base_url=self.base_url,
                proxy=self.proxy or None,
                timeout=self.timeout,
            )
            return self._normalize_results(raw, count)
        except Exception as e:
            print(f"[googletextsearch] google search failed: {e}")
            return []

    def _jina_request(self, url: str) -> str:
        fetch_url = self.jina_base_url.rstrip("/") + "/" + url.lstrip("/")
        resp = requests.get(
            fetch_url,
            proxies=_proxies(self.jina_proxy),
            timeout=self.timeout + 10,
        )
        resp.raise_for_status()
        return resp.text or ""

    def _fetch_page(self, url: str) -> str:
        for _ in range(self.max_retries):
            try:
                content = self._jina_request(url)
                if not content.strip():
                    continue
                content = re.sub(r"\n{2,}", "\n", content).strip()
                if len(re.sub(r"\s+", "", content)) < self.min_useful_chars:
                    return ""
                return content[: self.page_max_chars]
            except Exception as e:
                if "Client Error" in str(e):
                    break
        return ""

    def _rank1_from_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for item in results[: self.rank_fallback_max + 1]:
            url = item.get("url", "")
            if not url or _is_bad_domain(url, self.jina_bad_domains):
                continue
            page_text = self._fetch_page(url)
            if not page_text:
                continue
            try:
                summary = self.reader.read(query, page_text)
            except Exception as e:
                summary = f"[reader_error] {type(e).__name__}: {e}"
            if summary and not summary.startswith("[reader_error]"):
                if summary.strip() != "NO_RELEVANT_INFO":
                    content = summary[: self.rank1_max_chars]
                    return {
                        "ok": True,
                        "url": url,
                        "title": item.get("title", ""),
                        "content": content,
                        "truncated": len(summary) > self.rank1_max_chars,
                        "picked_rank": item.get("rank", 1),
                        "summary_mode": "reader",
                    }

        snippet_pool: List[str] = []
        for item in results[:5]:
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            if title or snippet:
                snippet_pool.append(f"[{item.get('rank')}] {title}\n{url}\n{snippet}")

        if not snippet_pool:
            return None

        combined = "\n\n".join(snippet_pool)
        try:
            summary = self.reader.read(query, combined)
        except Exception as e:
            summary = f"[reader_error] {type(e).__name__}: {e}"

        if summary and not summary.startswith("[reader_error]") and summary.strip() != "NO_RELEVANT_INFO":
            content = summary[: self.rank1_max_chars]
            mode = "snippet_reader"
        else:
            content = combined[: self.rank1_max_chars]
            mode = "snippet_raw"

        return {
            "ok": True,
            "url": results[0].get("url", ""),
            "title": results[0].get("title", ""),
            "content": content,
            "truncated": len(content) > self.rank1_max_chars,
            "picked_rank": 0,
            "summary_mode": mode,
        }

    def _search_single(
        self,
        query: str,
        count: int,
        fetch_rank1: bool = True,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return self._error(
                query,
                "missing SERPER_API_KEY; set it in scrpits/run.sh",
            )

        results = self._search_google(query, count)
        if not results:
            return self._error(query, "Google search returned no results.")

        rank1: Optional[Dict[str, Any]] = None
        if fetch_rank1:
            rank1 = self._rank1_from_results(query, results)

        return {
            "ok": True,
            "tool": "text_search",
            "query": query,
            "count": len(results),
            "results": results,
            "rank1": rank1,
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return _json_dumps([])

        if not isinstance(params, dict):
            return _json_dumps([])

        query = params.get("query")
        if not query:
            return _json_dumps([])

        if isinstance(query, str):
            query = [query]

        count = int(params.get("count", 10))
        fetch_rank1 = bool(params.get("fetch_rank1", True))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            responses = list(
                executor.map(
                    lambda q: self._search_single(q, count=count, fetch_rank1=fetch_rank1),
                    query,
                )
            )
        return _json_dumps(responses)


if __name__ == "__main__":
    tool = TextSearch()
    print(tool.call({"query": ["EMU3.5", "中国人民大学"]}))
