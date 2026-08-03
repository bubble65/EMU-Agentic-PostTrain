import os
import io
import json
import uuid
import base64
import requests
from datetime import datetime
from typing import Union, Optional, List

from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DRAW_OUTPUT_DIR = os.getenv(
    "DRAW_OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "tool_resp_sft_en"),
)
GEMINI_API_KEY_DEFAULT = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL_DEFAULT = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-3-pro-image")
GEMINI_ASPECT_RATIO_DEFAULT = os.getenv("GEMINI_ASPECT_RATIO", "4:3")
GEMINI_IMAGE_SIZE_DEFAULT = os.getenv("GEMINI_IMAGE_SIZE", "1K")
GEMINI_RESPONSE_MIME_TYPE_DEFAULT = os.getenv("GEMINI_RESPONSE_MIME_TYPE", "image/png")
GEMINI_TIMEOUT_DEFAULT = int(os.getenv("GEMINI_TIMEOUT", "180"))


@register_tool("draw", allow_overwrite=True)
class Draw(BaseTool):
    name = "draw"
    description = (
        "Generate an image from text via Gemini, optionally using reference images. "
        "Reference images are passed as a single `images` list in the desired order; "
        "each item can be either a local path under the project draw output directory or a remote URL. "
        "The prompt MUST refer to reference images strictly with the tokens "
        "[IMAGE1], [IMAGE2], [IMAGE3], ... (1-indexed, matching the order of `images`). "
        "No other natural-language reference is allowed (do not use 'Image 1', "
        "'the first image', '第1张图', '第一张参考图', etc.). "
        "The tool returns the saved image path."
    )

    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The image generation prompt. "
                    "To refer to a reference image, you MUST use the exact tokens "
                    "[IMAGE1], [IMAGE2], [IMAGE3], ... (1-indexed, matching the order "
                    "of `images`). Any other phrasing is forbidden — do NOT write "
                    "'Image 1', 'the first image', '第1张图', '第一张参考图', etc. "
                    "Example: '把 [IMAGE1] 里的人物贴到 [IMAGE2] 的背景上'."
                )
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Reference images, in the order they should be fed to the model. "
                    "Each item is either a local path under the project draw output directory, "
                    "or a remote URL (anything starting with http:// or https://)."
                )
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the generated image. Defaults to the project draw output directory."
            },
            "return_base64": {
                "type": "boolean",
                "description": "Whether to return the generated image base64 string.",
                "default": False
            },
        },
        "required": ["prompt"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}

        self.api_key = str(cfg.get("gemini_api_key") or GEMINI_API_KEY_DEFAULT)
        self.base_url = str(cfg.get("gemini_base_url") or GEMINI_BASE_URL_DEFAULT)
        self.model = str(cfg.get("gemini_model") or GEMINI_MODEL_DEFAULT)
        self.aspect_ratio = str(
            cfg.get("gemini_aspect_ratio") or GEMINI_ASPECT_RATIO_DEFAULT
        )
        self.image_size = str(
            cfg.get("gemini_image_size") or GEMINI_IMAGE_SIZE_DEFAULT
        )
        self.response_mime_type = str(
            cfg.get("gemini_response_mime_type") or GEMINI_RESPONSE_MIME_TYPE_DEFAULT
        )
        self.timeout = int(cfg.get("timeout", GEMINI_TIMEOUT_DEFAULT))

    def _build_output_path(self, output_path: Optional[str] = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uid = uuid.uuid4().hex[:8]

        default_dir = DEFAULT_DRAW_OUTPUT_DIR

        if not output_path:
            os.makedirs(default_dir, exist_ok=True)
            return os.path.join(default_dir, f"gen_{ts}_{uid}.png")

        if os.path.isdir(output_path) or output_path.endswith("/"):
            os.makedirs(output_path, exist_ok=True)
            return os.path.join(output_path, f"gen_{ts}_{uid}.png")

        dir_name = os.path.dirname(output_path) or "."
        os.makedirs(dir_name, exist_ok=True)

        base_name = os.path.basename(output_path)
        name, ext = os.path.splitext(base_name)
        if not ext:
            ext = ".png"

        return os.path.join(dir_name, f"{name}_{ts}_{uid}{ext}")

    def _normalize_to_png_bytes(self, raw: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.thumbnail((2048, 2048))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _load_ref_parts(
        self,
        images: Optional[List[str]] = None,
    ) -> List[dict]:
        parts = []

        for item in images or []:
            if item.startswith(("http://", "https://")):
                resp = requests.get(item, timeout=30)
                resp.raise_for_status()
                raw = resp.content
            else:
                with open(item, "rb") as f:
                    raw = f.read()

            png_bytes = self._normalize_to_png_bytes(raw)
            parts.append({
                "type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("utf-8"),
            })

        return parts

    def _build_multi_ref_prompt(self, prompt: str, num_images: int) -> str:
        if num_images <= 0:
            return prompt

        image_desc = "\n".join(
            [
                f"- [IMAGE{i + 1}]: The {i + 1}-th reference image in the upload order."
                for i in range(num_images)
            ]
        )

        return f"""You will receive {num_images} reference images. Please strictly follow the upload order and map them one-to-one with the numbered tokens below:

{image_desc}

The user's request will only refer to the reference images using fixed tokens like [IMAGE1], [IMAGE2], [IMAGE3], etc.
Please interpret each [IMAGEk] that appears in the prompt as the reference image with the corresponding number, and complete the generation or editing accordingly.

User request:
{prompt}
"""

    def _save_bytes(self, data: bytes, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path

    def _extract_output_image(self, result: dict):
        output_image = result.get("output_image")
        if isinstance(output_image, dict):
            data = output_image.get("data")
            if data:
                return data, output_image.get("mime_type") or self.response_mime_type

        steps = result.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                content = step.get("content") or []
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    data = part.get("data")
                    if data:
                        return data, part.get("mime_type") or self.response_mime_type

        if isinstance(result.get("data"), str) and result["data"]:
            return result["data"], self.response_mime_type

        return None, self.response_mime_type

    def call(self, params: Union[str, dict], **kwargs):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return json.dumps({
                    "ok": False,
                    "tool": "draw",
                    "error": "[Draw] Invalid params: must be a JSON object string."
                }, ensure_ascii=False)

        if not isinstance(params, dict):
            return json.dumps({
                "ok": False,
                "tool": "draw",
                "error": "[Draw] Invalid params: must be a dict."
            }, ensure_ascii=False)

        prompt = params.get("prompt")
        if not prompt:
            return json.dumps({
                "ok": False,
                "tool": "draw",
                "error": "[Draw] Missing required field: prompt."
            }, ensure_ascii=False)

        raw_output_path = params.get("output_path")
        output_path = self._build_output_path(raw_output_path)

        return_base64 = params.get("return_base64", False)

        images = list(params.get("images") or [])

        try:
            if not self.api_key:
                return json.dumps({
                    "ok": False,
                    "tool": "draw",
                    "error": "[Draw] Missing GEMINI_API_KEY; set it in scrpits/run.sh"
                }, ensure_ascii=False)

            ref_parts = self._load_ref_parts(images=images)

            if ref_parts:
                final_prompt = self._build_multi_ref_prompt(
                    prompt=prompt,
                    num_images=len(ref_parts),
                )
                mode = "edit_with_references"
            else:
                final_prompt = prompt
                mode = "text_to_image"

            parts = [{"type": "text", "text": final_prompt}] + ref_parts

            url = f"{self.base_url.rstrip('/')}/interactions"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }
            body = {
                "model": self.model,
                "input": parts,
                "response_format": {
                    "type": "image",
                    "mime_type": self.response_mime_type,
                    "aspect_ratio": self.aspect_ratio,
                    "image_size": self.image_size,
                },
            }

            resp = requests.post(url, json=body, headers=headers, timeout=self.timeout)

            try:
                result = resp.json()
            except Exception:
                return json.dumps({
                    "ok": False,
                    "tool": "draw",
                    "error": (
                        f"[Draw] Non-JSON response. "
                        f"status={resp.status_code}, text={resp.text[:1000]}"
                    )
                }, ensure_ascii=False)

            if resp.status_code >= 400:
                return json.dumps({
                    "ok": False,
                    "tool": "draw",
                    "error": (
                        f"[Draw] Request failed. "
                        f"status={resp.status_code}, response={result}"
                    )
                }, ensure_ascii=False)

            b64_str, response_mime_type = self._extract_output_image(result)

            if not b64_str:
                return json.dumps({
                    "ok": False,
                    "tool": "draw",
                    "error": f"[Draw] No image data in response: {result}"
                }, ensure_ascii=False)

            if b64_str.startswith("data:") and "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_str)
            saved_path = self._save_bytes(img_bytes, output_path)

            output = {
                "ok": True,
                "tool": "draw",
                "mode": mode,
                "saved_path": saved_path,
                "mime_type": response_mime_type,
            }

            if return_base64:
                output["base64"] = b64_str

            return json.dumps(output, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "ok": False,
                "tool": "draw",
                "error": f"[Draw] Exception: {repr(e)}"
            }, ensure_ascii=False)


if __name__ == "__main__":
    import rich
    tool = Draw()
    result2 = tool.call({
  "prompt": "In a vibrant neon-lit professional recording studio, [IMAGE1] (Fred again..), [IMAGE2] (Kieran Hebden/Four Tet) and [IMAGE3] (Sonny Moore/Skrillex) are huddled over the mixing console from [IMAGE4]. [IMAGE3] is adjusting a slider on the console with his hand, while [IMAGE1] and [IMAGE2] are listening intently to playback through shared over-ear studio headphones. The scene is candid, photorealistic, high detail, warm vibrant ambient studio lighting, natural relaxed expressions, 8K resolution",
  "images": [
    "https://www.atlanticrecords.com/sites/g/files/g2000015596/files/styles/artist_image_detail/public/2024-07/Fred%20again..%20ten%20Press%20Photo_0.jpg?itok=vW1bM8sb",
    "https://i.guim.co.uk/img/static/sys-images/Guardian/Pix/pictures/2015/7/10/1436531251032/1c679f39-a5cb-4425-b8ec-162f11554b86-2060x1236.jpeg?width=700&quality=85&auto=format&fit=max&s=13c8584003e91704f1f628179da83669",
    "https://external-preview.redd.it/sonny-moore-skrillex-officially-rejoins-from-first-to-last-v0-cfSgb39jFfehio4-JNUk3shFJ8QiY6iulXkpTC8c19o.jpg?auto=webp&s=1cd0236fef17301de7282553185b727ba029a9e3",
    "https://thumbs.dreamstime.com/b/vibrant-digital-audio-mixing-console-colorful-display-screens-high-tech-displays-set-professional-studio-environment-341143333.jpg"
  ]
})
    rich.print(result2)
