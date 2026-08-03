import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from tool_reader import DocReader

SERPER_API_KEY_DEFAULT = os.getenv("SERPER_API_KEY", "")
SERPER_IMAGES_BASE_URL_DEFAULT = os.getenv(
    "SERPER_IMAGES_BASE_URL", "https://google.serper.dev/images"
)
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

_PROBE_HEADERS = {"User-Agent": "Wget/1.21.2"}


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


def _url_downloadable(url: str, timeout: float = 5.0) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False

    def _ct_ok(resp: requests.Response) -> bool:
        ct = (resp.headers.get("Content-Type") or "").lower()
        if not ct:
            return True
        return ct.startswith("image/") or "octet-stream" in ct

    try:
        resp = requests.head(
            url,
            headers=_PROBE_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.ok and _ct_ok(resp):
            return True
    except Exception:
        pass

    try:
        resp = requests.get(
            url,
            headers=_PROBE_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        try:
            if not resp.ok or not _ct_ok(resp):
                return False
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    return True
            return False
        finally:
            resp.close()
    except Exception:
        return False


def search_google_images(
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
        base_url or SERPER_IMAGES_BASE_URL_DEFAULT,
        headers={
            "X-API-KEY": resolved_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": topk, "gl": "us", "hl": "en"},
        proxies=_proxies(proxy) if proxy is not None else None,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("images", []) or []


@register_tool("image_search", allow_overwrite=True)
class ImageSearch(BaseTool):
    name = "image_search"
    description = "Image search with public Jina page fetch."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "图片搜索词数组。可以一次传多个互补查询。",
            },
            "count": {
                "type": "integer",
                "description": "每个 query 返回图片数量，默认使用工具初始化 count。",
                "default": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg)
        cfg = cfg or {}

        self.api_key: str = cfg.get("serper_api_key", SERPER_API_KEY_DEFAULT)
        self.base_url: str = cfg.get(
            "serper_images_base_url", SERPER_IMAGES_BASE_URL_DEFAULT
        )
        self.proxy: str = cfg.get("serper_proxy", "")
        self.timeout: int = int(cfg.get("timeout", 100))
        self.max_retries: int = int(cfg.get("max_retries", 3))
        self.max_workers: int = int(cfg.get("max_workers", 3))
        self.probe_timeout: float = float(cfg.get("probe_timeout", 5.0))
        self.verify_page_relevance: bool = bool(
            cfg.get("verify_page_relevance", True)
        )
        self.relevance_scan_limit: int = int(cfg.get("relevance_scan_limit", 15))
        self.rank_fallback_max: int = int(cfg.get("rank_fallback_max", 4))
        self.rank1_max_chars: int = int(cfg.get("rank1_max_chars", 1000))
        self.page_max_chars: int = int(cfg.get("page_max_chars", 128000))
        self.min_useful_chars: int = int(
            cfg.get("min_useful_chars", MIN_USEFUL_CHARS_DEFAULT)
        )
        self.jina_base_url: str = cfg.get("jina_base_url", JINA_BASE_URL_DEFAULT)
        self.jina_proxy: str = cfg.get("jina_proxy", JINA_PROXY_DEFAULT)
        self.jina_bad_domains: List[str] = list(
            cfg.get("jina_bad_domains", JINA_BAD_DOMAINS_DEFAULT)
        )
        self.reader: DocReader = cfg.get("reader") or DocReader()

    def _error(self, query: str, error: str = "") -> Dict[str, Any]:
        return {
            "ok": False,
            "tool": "image_search",
            "query": query,
            "count": 0,
            "results": [],
            "error": error,
        }

    def _search_google_images(self, query: str, count: int) -> List[Dict[str, Any]]:
        try:
            return search_google_images(
                query,
                topk=count,
                api_key=self.api_key,
                base_url=self.base_url,
                proxy=self.proxy or None,
                timeout=self.timeout,
            )
        except Exception as e:
            print(f"[googleimagesearch] google images search failed: {e}")
            return []

    def _choose_image_url(self, image_url: str, thumbnail_url: str) -> str:
        for candidate in (image_url, thumbnail_url):
            if candidate and not candidate.startswith("data:"):
                return candidate
        return ""

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

    def _page_relevant(
        self, query: str, source_page_url: str
    ) -> Tuple[Optional[str], bool]:
        summary = None
        if not source_page_url:
            return summary, False
        if _is_bad_domain(source_page_url, self.jina_bad_domains):
            return summary, False

        page_text = self._fetch_page(source_page_url)
        if not page_text:
            return summary, False

        try:
            summary = self.reader.read(query, page_text)
        except Exception as e:
            print(f"[googleimagesearch] reader error on {source_page_url[:120]}: {e}")
            return summary, False

        if not summary:
            return summary, False
        if summary.startswith("[reader_error]"):
            return summary, False
        if summary.strip() == "NO_RELEVANT_INFO":
            return summary, False
        return summary, True

    def _normalize_results(
        self,
        raw: List[Dict[str, Any]],
        count: int,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        skipped_undownloadable = 0
        skipped_irrelevant = 0
        do_relevance = self.verify_page_relevance and bool(query)

        for idx, item in enumerate(raw, 1):
            if len(results) >= count:
                break
            if do_relevance and idx > self.relevance_scan_limit:
                break

            image_url = (item.get("imageUrl") or "").strip()
            thumbnail_url = (item.get("thumbnailUrl") or "").strip()
            url = self._choose_image_url(image_url, thumbnail_url)
            if not url:
                continue

            if not _url_downloadable(url, timeout=self.probe_timeout):
                alt = thumbnail_url if url != thumbnail_url else ""
                if alt and _url_downloadable(alt, timeout=self.probe_timeout):
                    url = alt
                else:
                    skipped_undownloadable += 1
                    continue

            source_page_url = item.get("link", "") or item.get("source", "")
            summary = None
            if do_relevance:
                summary, is_relevant = self._page_relevant(query, source_page_url)
                if not is_relevant:
                    skipped_irrelevant += 1
                    continue

            results.append(
                {
                    "summary": summary,
                    "rank": len(results) + 1,
                    "title": item.get("title", "") or "",
                    "url": url,
                    "source_page_url": source_page_url,
                    "image_url": image_url,
                    "thumbnail_url": thumbnail_url,
                    "image_width": item.get("imageWidth"),
                    "image_height": item.get("imageHeight"),
                    "thumbnail_width": item.get("thumbnailWidth"),
                    "thumbnail_height": item.get("thumbnailHeight"),
                    "domain": item.get("domain", ""),
                    "source": item.get("source", ""),
                    "google_url": item.get("googleUrl", ""),
                    "position": item.get("position"),
                }
            )

        if skipped_undownloadable or skipped_irrelevant:
            print(
                f"[googleimagesearch] verified {len(results)}/{count} urls "
                f"(skipped {skipped_undownloadable} undownloadable, "
                f"{skipped_irrelevant} irrelevant)"
            )
        return results

    def _google_image_search(
        self, query: str, count: int
    ) -> List[Dict[str, Any]]:
        for _ in range(self.max_retries):
            try:
                raw = self._search_google_images(query, count)
                if raw:
                    normalized = self._normalize_results(raw, count, query=query)
                    if normalized:
                        return normalized
                break
            except Exception as e:
                print(f"[googleimagesearch] attempt failed: {e}")
        return []

    def _search_single(self, query: str, count: int) -> Dict[str, Any]:
        if not self.api_key:
            return self._error(
                query,
                "missing SERPER_API_KEY; set it in scrpits/run.sh",
            )

        results = self._google_image_search(query, count)
        if not results:
            return self._error(query, f"No image results found for query: {query}")
        return {
            "ok": True,
            "tool": "image_search",
            "query": query,
            "count": len(results),
            "results": results,
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

        count = int(params.get("count", 1))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            responses = list(
                executor.map(lambda q: self._search_single(q, count=count), query)
            )

        return _json_dumps(responses)


if __name__ == "__main__":
    tool = ImageSearch()
    print(tool.call({"query": ["中国人民大学 徐君教授", "EMU3.5"]}))
