"""Unified VLM client: vLLM local, vLLM HTTP server, or Gemini API."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

_VLLM_HTTP_BACKENDS = frozenset({"vllm", "openai", "qwen"})
_LOCAL_BACKENDS = frozenset({"local", "vllm-local"})


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def is_local_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "local")).lower()
    return b in _LOCAL_BACKENDS


def is_vllm_http_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "local")).lower()
    return b in _VLLM_HTTP_BACKENDS


def require_vllm_server(base_url: str | None = None, timeout: float = 10.0) -> str:
    """Verify vLLM OpenAI server is up; raise ConnectionError with setup hint if not."""
    url = (base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    if not url:
        raise ValueError(
            "Set VLM_BASE_URL in .env (e.g. http://127.0.0.1:8000/v1). "
            "Or use --vlm-backend local / run_pose_vlm_eval.py (no server)."
        )
    models_url = f"{url}/models"
    try:
        import httpx

        resp = httpx.get(models_url, timeout=timeout)
        resp.raise_for_status()
    except ImportError as e:
        raise RuntimeError("Install httpx (pip install httpx) for server health checks") from e
    except Exception as e:
        raise ConnectionError(
            f"vLLM HTTP server not reachable at {url} ({e}). "
            "Use in-process inference instead:\n"
            "  python adhoc/generation/robotarm/run_pose_vlm_eval.py --experiment multitile20"
        ) from e
    return url


class VLMClient:
    """Text + optional PIL images → model text response."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("VLM_BACKEND", "local")).lower()
        self.model = model or os.getenv("VLM_MODEL") or self._default_model()
        self._client: Any = None
        self._kind = self.backend

        if self.backend in _LOCAL_BACKENDS:
            self._kind = "local"
            from vllm_local import get_vllm_engine

            self._client = get_vllm_engine(model=self.model)
        elif self.backend in _VLLM_HTTP_BACKENDS:
            self._kind = "vllm_http"
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLM_API_KEY") or "EMPTY"
            url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if not url:
                raise ValueError(
                    "Set VLM_BASE_URL for HTTP vllm backend, "
                    "or use --vlm-backend local for in-process inference."
                )
            self._client = OpenAI(api_key=key, base_url=url)
        elif self.backend == "gemini":
            from google import genai

            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for gemini backend")
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unknown VLM backend: {self.backend} (use local, vllm, or gemini)")

    def _default_model(self) -> str:
        if self.backend in _LOCAL_BACKENDS or self.backend in _VLLM_HTTP_BACKENDS:
            return os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        return os.getenv("VLM_MODEL", "gemini-2.5-pro")

    def generate(self, prompt: str, images: list[Image.Image] | None = None) -> str:
        images = images or []
        if self._kind == "local":
            return self._client.generate(prompt, images)

        if self._kind == "vllm_http":
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(img)}"},
                    }
                )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": parts}],
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()

        contents: list[Any] = [prompt]
        contents.extend(images)
        resp = self._client.models.generate_content(model=self.model, contents=contents)
        return (resp.text or "").strip()
