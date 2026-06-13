"""Unified VLM client: transformers / vLLM local / HTTP / Gemini."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

_VLLM_HTTP_BACKENDS = frozenset({"vllm", "openai", "qwen"})
_VLLM_LOCAL_BACKENDS = frozenset({"local", "vllm-local"})
_TRANSFORMERS_BACKENDS = frozenset({"transformers", "hf"})
_INPROCESS_BACKENDS = _VLLM_LOCAL_BACKENDS | _TRANSFORMERS_BACKENDS


def resolve_vlm_backend(args: Any | None = None, *, default: str = "transformers") -> str:
    """Resolve backend from argparse Namespace, env, or explicit string."""
    if isinstance(args, str):
        return args.strip().lower() or default
    raw = (
        getattr(args, "vlm_backend", None)
        if args is not None
        else None
    ) or os.getenv("VLM_BACKEND") or default
    return str(raw).strip().lower() or default


def setup_vlm_from_args(args: Any) -> tuple[str, "VLMClient | None"]:
    """Return (backend, client). Gemini leaves client None (API used separately)."""
    backend = resolve_vlm_backend(args)
    shared = getattr(args, "vlm", None)
    if shared is not None:
        return backend, shared
    if backend == "gemini":
        return backend, None
    if is_vllm_http_backend(backend):
        require_vllm_server()
    elif is_inprocess_backend(backend):
        init_inprocess_engine(backend, getattr(args, "model", None))
    model = getattr(args, "model", None)
    return backend, VLMClient(backend=backend, model=model)


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def is_inprocess_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
    return b in _INPROCESS_BACKENDS


def is_local_backend(backend: str | None = None) -> bool:
    return is_inprocess_backend(backend)


def is_transformers_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
    return b in _TRANSFORMERS_BACKENDS


def is_vllm_local_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
    return b in _VLLM_LOCAL_BACKENDS


def vlm_batch_size(backend: str | None = None) -> int:
    """In-process vLLM batch size (default 10); 1 for other backends."""
    if is_vllm_local_backend(backend):
        return max(1, int(os.getenv("VLM_BATCH_SIZE", "10")))
    return 1


def is_vllm_http_backend(backend: str | None = None) -> bool:
    b = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
    return b in _VLLM_HTTP_BACKENDS


def init_inprocess_engine(backend: str | None = None, model: str | None = None) -> None:
    """Load model once for in-process backends."""
    from hf_cache_setup import setup_hf_cache

    setup_hf_cache(os.environ.get("HF_HOME"))
    b = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
    if b in _TRANSFORMERS_BACKENDS:
        from transformers_local import get_transformers_engine

        get_transformers_engine(model=model)
    elif b in _VLLM_LOCAL_BACKENDS:
        from vllm_local import get_vllm_engine

        get_vllm_engine(model=model)
    else:
        raise ValueError(f"Not an in-process backend: {b}")


def require_vllm_server(base_url: str | None = None, timeout: float = 10.0) -> str:
    """Verify vLLM OpenAI server is up; raise ConnectionError with setup hint if not."""
    url = (base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    if not url:
        raise ValueError("Set VLM_BASE_URL for HTTP vllm backend.")
    models_url = f"{url}/models"
    try:
        import httpx

        resp = httpx.get(models_url, timeout=timeout)
        resp.raise_for_status()
    except ImportError as e:
        raise RuntimeError("Install httpx for server health checks") from e
    except Exception as e:
        raise ConnectionError(f"vLLM HTTP server not reachable at {url} ({e})") from e
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
        self.backend = (backend or os.getenv("VLM_BACKEND", "transformers")).lower()
        self.model = model or os.getenv("VLM_MODEL") or self._default_model()
        self._client: Any = None
        self._kind = self.backend

        if self.backend in _TRANSFORMERS_BACKENDS:
            self._kind = "transformers"
            from transformers_local import get_transformers_engine

            self._client = get_transformers_engine(model=self.model)
        elif self.backend in _VLLM_LOCAL_BACKENDS:
            self._kind = "vllm_local"
            from vllm_local import get_vllm_engine

            self._client = get_vllm_engine(model=self.model)
        elif self.backend in _VLLM_HTTP_BACKENDS:
            self._kind = "vllm_http"
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLM_API_KEY") or "EMPTY"
            url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if not url:
                raise ValueError("Set VLM_BASE_URL for HTTP vllm backend.")
            self._client = OpenAI(api_key=key, base_url=url)
        elif self.backend == "gemini":
            from google import genai

            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GOOGLE_API_KEY for gemini backend")
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unknown VLM backend: {self.backend}")

    def _default_model(self) -> str:
        if self.backend in _INPROCESS_BACKENDS or self.backend in _VLLM_HTTP_BACKENDS:
            return os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        return os.getenv("VLM_MODEL", "gemini-2.5-pro")

    def generate(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        videos: list[str] | None = None,
    ) -> str:
        images = images or []
        videos = videos or []
        if self._kind in {"transformers", "vllm_local"}:
            return self._client.generate(prompt, images, videos=videos)

        if self._kind == "vllm_http":
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(img)}"},
                    }
                )
            from vlm_sampling import sampling_temperature

            temp = sampling_temperature()
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": parts}],
                temperature=max(0.0, temp),
            )
            return (resp.choices[0].message.content or "").strip()

        contents: list[Any] = [prompt]
        contents.extend(images)
        resp = self._client.models.generate_content(model=self.model, contents=contents)
        return (resp.text or "").strip()

    def generate_many(
        self,
        requests: list[dict[str, Any]],
    ) -> list[str]:
        """Batch generate when vLLM local supports it; else sequential."""
        if not requests:
            return []
        if self._kind == "vllm_local" and len(requests) > 1:
            return self._client.generate_batch(requests)
        return [
            self.generate(
                r["prompt"],
                images=r.get("images"),
                videos=r.get("videos"),
            )
            for r in requests
        ]
