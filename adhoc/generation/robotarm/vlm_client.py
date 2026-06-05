"""Unified VLM client: Gemini API or OpenAI-compatible server (vLLM / Qwen-VL)."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
        self.backend = (backend or os.getenv("VLM_BACKEND", "gemini")).lower()
        self.model = model or os.getenv("VLM_MODEL") or self._default_model()
        self._client: Any = None
        self._kind = self.backend

        if self.backend in {"openai", "qwen", "vllm"}:
            self._kind = "openai"
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLM_API_KEY") or "EMPTY"
            url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if not url:
                raise ValueError("Set VLM_BASE_URL for openai/qwen/vllm backend (e.g. http://host:8000/v1)")
            self._client = OpenAI(api_key=key, base_url=url)
        elif self.backend == "gemini":
            from google import genai

            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for gemini backend")
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unknown VLM backend: {self.backend}")

    def _default_model(self) -> str:
        if self.backend in {"openai", "qwen", "vllm"}:
            return os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        return os.getenv("VLM_MODEL", "gemini-2.5-pro")

    def generate(self, prompt: str, images: list[Image.Image] | None = None) -> str:
        images = images or []
        if self._kind == "openai":
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
