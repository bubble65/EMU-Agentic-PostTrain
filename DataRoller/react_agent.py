import json
import os
import base64
import mimetypes
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import requests
import tiktoken
import rich
from openai import OpenAI
from PIL import Image
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool


MAX_LLM_CALL_PER_RUN = 8
MAX_TOKEN_LENGTH = 32768 * 10
DOUBAO_BASE_URL_DEFAULT = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_MODEL_DEFAULT = os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215")
DOUBAO_API_KEY_DEFAULT = os.getenv("ARK_API_KEY", "")
DOUBAO_MAX_OUTPUT_TOKENS_DEFAULT = 4096
DOUBAO_TEMPERATURE_DEFAULT = 0.0
DOUBAO_TOP_P_DEFAULT = 1.0

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS

IMG_DOWNGRADE_MAX_SIDE = 1024
IMG_DOWNGRADE_JPEG_QUALITY = 80
IMG_DOWNGRADE_FETCH_TIMEOUT = 15


# ---------- 图片处理 ----------

def build_mm_user_message(text: str, image_urls=None):
    """构造多模态 user 消息：先放图片，再放文字。
    内容块格式如下：
      - 文本: {"type": "text", "text": ...}
      - 图片: {"type": "image_url", "image_url": {"url": ...}}
    """
    image_urls = image_urls or []
    content = []
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    content.append({
        "type": "text",
        "text": text,
    })
    return {"role": "user", "content": content}


def _download_image_as_data_url(url: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        b64 = base64.b64encode(resp.content).decode()
        return f"data:{content_type};base64,{b64}"
    except Exception as e:
        print(f"[WARN] Failed to download image: {url}, {e}")
        return None


def _load_local_image_as_data_url(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        media_type, _ = mimetypes.guess_type(path)
        if not media_type or not media_type.startswith("image/"):
            media_type = "image/png"
        b64 = base64.b64encode(data).decode()
        return f"data:{media_type};base64,{b64}"
    except Exception as e:
        print(f"[WARN] Failed to load local image: {path}, {e}")
        return None


def _fetch_and_compress_to_b64(url_or_path: str,
                               max_side: int = IMG_DOWNGRADE_MAX_SIDE,
                               quality: int = IMG_DOWNGRADE_JPEG_QUALITY) -> Optional[str]:
    """本地抓取（URL 或本地路径）→ PIL 解码 → 缩放 → 重新编码成 JPEG → base64 data URI。
    用作后端抓不到远程 URL 时的兜底，避免 413：长边压到 max_side，JPEG q=quality，单图通常几百 KB。
    任何异常返回 None（调用方据此丢弃这张图）。
    """
    try:
        if url_or_path.startswith(("http://", "https://")):
            r = requests.get(url_or_path, timeout=IMG_DOWNGRADE_FETCH_TIMEOUT)
            r.raise_for_status()
            raw = r.content
        else:
            with open(url_or_path, "rb") as f:
                raw = f.read()

        img = Image.open(BytesIO(raw))
        # JPEG 不支持 alpha / palette，统一转 RGB
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # 长边超 max_side 才缩，反之原样
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            img = img.resize((int(w * scale), int(h * scale)), _LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[WARN] _fetch_and_compress_to_b64 failed: {url_or_path}, {e}")
        return None


def _downgrade_image_urls_to_b64(messages: List[dict]) -> int:
    """就地把 messages 里所有 image_url 块的远程 URL 换成压缩后的 base64 data URI。
    - 已经是 data: URI 的跳过（不会重复处理 draw 产物）。
    - 远程抓 + 压缩失败的图块直接从 content 里删除（避免再触发 400/413）。
    返回处理过的图块数（包含被丢弃的）。
    """
    handled = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_blocks = []
        for blk in content:
            if not isinstance(blk, dict) or blk.get("type") != "image_url":
                new_blocks.append(blk)
                continue
            iu = blk.get("image_url")
            url = iu.get("url", "") if isinstance(iu, dict) else (iu or "")
            if not url:
                # 空的图块，删掉
                handled += 1
                continue
            if url.startswith("data:"):
                # 已经是 base64，留着
                new_blocks.append(blk)
                continue
            # 远程 URL：本地抓 + 压缩
            data_url = _fetch_and_compress_to_b64(url)
            handled += 1
            if data_url:
                new_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            # else: 本地也抓不到，整块丢弃
        msg["content"] = new_blocks
    return handled


def _should_downgrade_image_error(error: Exception) -> bool:
    """判断这次异常是不是 image_url 抓取失败，适合降级成 base64 再重试。"""
    text = str(error).lower()
    if "image_url" in text:
        return True
    if "input_image" in text and ("invalid" in text or "download" in text):
        return True

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code == 400:
            body = ""
            try:
                body = response.text
            except Exception:
                body = ""
            body = (body or "").lower()
            if "image_url" in body or "download" in body or "invalidparameter" in body:
                return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body_text = json.dumps(body, ensure_ascii=False).lower()
        if "image_url" in body_text or "download" in body_text or "invalidparameter" in body_text:
            return True
    elif isinstance(body, str):
        body_text = body.lower()
        if "image_url" in body_text or "download" in body_text or "invalidparameter" in body_text:
            return True
    return False


def _text_from_value(value):
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
        parts = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts.append(_text_from_value(item.get("content")))
        text = "".join(parts)
        if text:
            return text

    return ""


# ---------- 干净的 tool_response 拼装 ----------

def _format_text_search_response(parsed) -> str:
    """text_search 工具返回 -> 「query : xxx\n查询结果 : title | content」格式。
    parsed 形如:
      [{"ok": true, "tool": "text_search", "query": "...", "rank1": {"title": "...", "content": "..."}}]
    """
    items = parsed if isinstance(parsed, list) else [parsed]
    lines: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query = item.get("query", "")
        rank1 = item.get("rank1") or {}
        if not item.get("ok"):
            err = item.get("error") or "search failed"
            lines.append(f"query : {query}\n查询结果 : [error] {err}")
            continue
        title = (rank1.get("title") or "").strip()
        content = (rank1.get("content") or "").strip()
        if not (title or content):
            lines.append(f"query : {query}\n查询结果 : [empty]")
            continue
        lines.append(f"query : {query}\n查询结果 : {title} | {content}")
    return "\n\n".join(lines)


def _format_image_search_response(
    parsed,
    image_counter_start: int,
) -> Tuple[str, List[str], List[str], int]:
    """image_search 工具返回 -> 「query : xxx\n查询结果 : <desp>...摘要 和 title</desp>[external image n]url[/external image n]」格式。

    返回:
      text:                拼成的工具响应文本（含 [external image n]...[/external image n] 标签）
      labels:              本次新分配的图片编号（与 image_urls 一一对应），用于回放给 model 的 vision 块
      image_urls:          本次新增的图片 URL 列表
      next_counter:        更新后的全局计数器（供下一轮 tool_response 继续累加）
    """
    items = parsed if isinstance(parsed, list) else [parsed]
    text_blocks: List[str] = []
    image_urls: List[str] = []
    labels: List[str] = []
    counter = image_counter_start

    for item in items:
        if not isinstance(item, dict):
            continue
        query = item.get("query", "")
        if not item.get("ok"):
            err = item.get("error") or "image search failed"
            text_blocks.append(f"query : {query}\n查询结果 : [error] {err}")
            continue
        results = item.get("results") or []
        if not results:
            text_blocks.append(f"query : {query}\n查询结果 : [empty]")
            continue

        per_result_parts: List[str] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "") or ""
            if not url:
                continue
            title = (r.get("title") or "").strip()
            summary = (r.get("summary") or "").strip()
            # 图片对应网页的文本摘要 和 title
            desp_inner = summary
            if title:
                desp_inner = f"{summary} 和 {title}" if summary else title

            n = counter
            counter += 1
            labels.append(str(n))
            image_urls.append(url)

            per_result_parts.append(
                f"<desp>{desp_inner}</desp>"
                f"[external image {n}]{url}[/external image {n}]"
            )

        if per_result_parts:
            text_blocks.append(
                f"query : {query}\n查询结果 : " + "".join(per_result_parts)
            )
        else:
            text_blocks.append(f"query : {query}\n查询结果 : [empty]")

    return "\n\n".join(text_blocks), labels, image_urls, counter


def _format_draw_response(
    parsed,
    image_counter_start: int,
) -> Tuple[str, List[str], List[str], int]:
    """draw 工具返回 -> 干净的工具响应文本 + 一张生成图（local data URI）。

    parsed 形如:
      {"ok": true, "tool": "draw", "saved_path": "...", "mode": "...", "mime_type": "image/png"}
    """
    if not isinstance(parsed, dict):
        return "查询结果 : [error] invalid draw response", [], [], image_counter_start

    if not parsed.get("ok"):
        err = parsed.get("error") or "draw failed"
        return f"查询结果 : [error] {err}", [], [], image_counter_start

    saved_path = parsed.get("saved_path") or ""
    data_url: Optional[str] = None
    if saved_path:
        data_url = _load_local_image_as_data_url(saved_path)
    if not data_url and parsed.get("base64"):
        data_url = f"data:image/png;base64,{parsed['base64']}"

    if not data_url:
        return "查询结果 : [error] no image produced", [], [], image_counter_start

    n = image_counter_start
    text = (
        f"查询结果 : <desp>生成图片已保存到 {saved_path}</desp>"
        f"[internal image {n}]{saved_path}[/internal image {n}]"
    )
    return text, [str(n)], [data_url], image_counter_start + 1


def _build_tool_response(
    tool_name: str,
    tool_result_str: str,
    image_counter_start: int,
) -> Tuple[str, List[str], List[str], int]:
    """工具结果 -> <tool_response>...</tool_response> 包好的干净文本。

    返回:
      wrapped_text:    <tool_response>\n ... \n</tool_response>
      labels:          本次产生的图片编号
      image_urls:      本次产生的图片 URL
      next_counter:    更新后的全局计数器
    """
    # 优先尝试当成 JSON 解析；解析失败就退化成原始字符串
    try:
        parsed = json.loads(tool_result_str)
    except (json.JSONDecodeError, TypeError):
        inner = tool_result_str.strip()
        return f"<tool_response>\n{inner}\n</tool_response>", [], [], image_counter_start

    labels: List[str] = []
    image_urls: List[str] = []
    counter = image_counter_start

    if tool_name == "text_search":
        inner = _format_text_search_response(parsed)
    elif tool_name == "image_search":
        inner, labels, image_urls, counter = _format_image_search_response(
            parsed, image_counter_start=counter
        )
    elif tool_name == "draw":
        inner, labels, image_urls, counter = _format_draw_response(
            parsed, image_counter_start=counter
        )
    else:
        # 未知工具：原样吐 JSON
        inner = json.dumps(parsed, ensure_ascii=False)

    wrapped = f"<tool_response>\n{inner}\n</tool_response>"
    return wrapped, labels, image_urls, counter


# ---------- Responses API 消息整形 ----------

def _to_messages_payload(msgs):
    """把内部对话历史拆成 (system_text, messages_for_api)。
    - role=system: 单独提取出来放到顶层系统提示
    - role=user/assistant: 文本块转成 {"type":"input_text"}，图片块转成 {"type":"input_image"}
    """
    system_text = ""
    out_messages = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                system_text = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                system_text = content or ""
            continue

        # user / assistant
        if isinstance(content, str):
            out_messages.append({"role": role, "content": content, "partial": False})
            continue

        if isinstance(content, list):
            blocks = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    blocks.append({"type": "input_text", "text": b.get("text", "")})
                elif t == "image_url":
                    iu = b.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url", "")
                    else:
                        url = iu or ""
                    if url:
                        blocks.append({"type": "input_image", "image_url": url, "detail": "auto"})
            out_messages.append({"role": role, "content": blocks, "partial": False})
        else:
            out_messages.append({"role": role, "content": "", "partial": False})

    return system_text, out_messages


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        llm = llm or {}
        self.llm_generate_cfg = llm.get("generate_cfg", {})
        self.llm_local_path = llm.get("model", DOUBAO_MODEL_DEFAULT)
        self.model = str(llm.get("model") or DOUBAO_MODEL_DEFAULT)
        self.api_key = str(llm.get("api_key") or DOUBAO_API_KEY_DEFAULT)
        self.base_url = str(llm.get("base_url") or DOUBAO_BASE_URL_DEFAULT)
        self.max_output_tokens = int(
            self.llm_generate_cfg.get("max_output_tokens", DOUBAO_MAX_OUTPUT_TOKENS_DEFAULT)
        )
        self.temperature = float(self.llm_generate_cfg.get("temperature", DOUBAO_TEMPERATURE_DEFAULT))
        self.top_p = float(self.llm_generate_cfg.get("top_p", DOUBAO_TOP_P_DEFAULT))
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def call_server(self, msgs, max_tries=10):
        """通过 Doubao/OpenAI-compatible responses API 调用大模型。
        - 输入消息使用 Responses API 的 message 结构
        - 文本/图片块分别转成 input_text / input_image
        - 返回结果优先取 output_text，再回退到完整响应解析
        - 后端抓不到 image_url 时，一次性把所有远程 URL 本地下载 + PIL 压缩成 base64 data URI，再重试；本地也抓不到的图块直接丢弃。
        """
        if self.client is None:
            if not self.api_key:
                return "Doubao server error: missing ARK_API_KEY"
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        already_downgraded = False  # 每次 call_server 最多降级一次，避免死循环

        for attempt in range(max_tries):
            try:
                system_text, messages_payload = _to_messages_payload(msgs)
                input_messages = []
                if system_text:
                    input_messages.append({"role": "system", "content": system_text, "partial": False})
                input_messages.extend(messages_payload)

                resp = self.client.responses.create(
                    model=self.model,
                    input=input_messages,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=300,
                    extra_body={
                        "thinking": {"type": "disabled"}, 
                    },
                )

                resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else {}
                text = _extract_text_from_resp(resp_dict) or getattr(resp, "output_text", "") or ""
                if text:
                    for stop in ("\n<tool_response>", "<tool_response>"):
                        idx = text.find(stop)
                        if idx >= 0:
                            text = text[:idx]
                            break
                    return text

                print(f"[Doubao] empty content on attempt {attempt + 1}: {resp_dict}")
            except Exception as e:
                if not already_downgraded and _should_downgrade_image_error(e):
                    print("[Doubao] image fetch failed, downgrading remote URLs to base64 and retrying")
                    n = _downgrade_image_urls_to_b64(msgs)
                    print(f"[Doubao] downgraded {n} image block(s) to local base64, retrying...")
                    already_downgraded = True
                    continue
                print(f"[Doubao] attempt {attempt + 1}/{max_tries} failed: {e}")
                if attempt == max_tries - 1:
                    return f"Doubao server error: {e}"
                continue

        return "server empty response"

    def count_tokens(self, messages, model="gpt-4o"):
        tokenizer = tiktoken.encoding_for_model(model)
        # 对于多模态消息，只统计文本部分的 token；图片用占位符粗估
        text_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        text_parts.append("<image_placeholder>")

        full_text = "\n".join(text_parts)
        return len(tokenizer.encode(full_text))

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        self.model = model
        try:
            question = data['item']['question']
        except Exception:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        answer = data['item']['answer']
        self.user_prompt = user_prompt + question

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_prompt},
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0

        # 整条 trace 维护一个图片计数器，[external image n] 的 n 在整条 trace 内单调递增
        image_counter = 1

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)

            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]

            messages.append({"role": "assistant", "content": content.strip()})

            rich.print(f'[Agent]content: {content}')

            # ── 工具调用 ──
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                # rich.print(f'tool_call: {tool_call_str}')

                tool_name = ""
                try:
                    tool_call = json.loads(tool_call_str)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self._call_tool(tool_name, tool_args)
                except Exception:
                    result = ('Error: Tool call is not a valid JSON. '
                              'Tool call must contain a valid "name" and "arguments" field.')

                # 拼成干净的 <tool_response>...</tool_response>
                wrapped_text, _labels, image_urls, image_counter = _build_tool_response(
                    tool_name, result, image_counter_start=image_counter,
                )

                if image_urls:
                    messages.append(build_mm_user_message(
                        text=wrapped_text,
                        image_urls=image_urls,
                    ))
                else:
                    messages.append({"role": "user", "content": wrapped_text})

            # ── 终止判断 ──
            if '<|box_start|>' in content and '<|box_end|>' in content:
                termination = 'answer'
                break

            if num_llm_calls_available <= 0 and '<|box_start|>' not in content:
                messages[-1] = {
                    "role": "user",
                    "content": 'Sorry, the number of llm calls exceeds the limit.',
                }

            # ── token 超限处理 ──
            max_tokens = MAX_TOKEN_LENGTH
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                messages[-1] = {
                    "role": "user",
                    "content": (
                        "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer in the "
                        "following format:<think>your final thinking</think>\n"
                        "<|box_start|>your answer<|box_end|>"
                    ),
                }
                content = self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<|box_start|>' in content and '<|box_end|>' in content:
                    prediction = content.split('<|box_start|>')[1].split('<|box_end|>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = content
                    termination = 'format error: generate an answer as token limit reached'
                return {
                    "question": question,
                    "answer": answer,
                    "rollout_id": data['rollout_id'],
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }

        if '<|box_start|>' in messages[-1].get('content', ''):
            last_content = messages[-1].get('content', '')
            if isinstance(last_content, list):
                last_content = ''.join(
                    b.get('text', '') for b in last_content
                    if isinstance(b, dict) and b.get('type') == 'text'
                )
            prediction = last_content.split('<|box_start|>')[1].split('<|box_end|>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        # 清理输出：把 user 消息里的 image_url 内容置空，避免落盘的 JSONL 太大
        for message in messages:
            if message.get('role') == 'user':
                content = message.get('content', '')
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'image_url':
                        block['image_url'] = ""

        return {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
        }
