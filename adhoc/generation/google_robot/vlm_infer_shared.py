"""Shared VLM inference helpers for Google Robot verify/compare scripts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image


def extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return __import__("json").loads(s)


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return extract_json(text)
    except Exception as e:
        return {"parse_error": str(e), "raw_text": text}


def gif_first_frame(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.seek(0)
        return im.convert("RGB")


def load_vlm_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".gif":
        return gif_first_frame(path)
    return Image.open(path).convert("RGB")


def vlm_generate_json(
    prompt: str,
    *,
    model: str,
    vlm: Any | None = None,
    client: Any | None = None,
    images: list[Image.Image] | None = None,
    videos: list[str] | None = None,
) -> dict[str, Any]:
    if vlm is not None:
        text = vlm.generate(prompt, images=images, videos=videos)
        return parse_json_response(text)
    if client is None:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini backend.")
    from google.genai import types

    parts: list[Any] = []
    for img in images or []:
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    for vpath in videos or []:
        parts.append(types.Part.from_bytes(data=Path(vpath).read_bytes(), mime_type="video/mp4"))
    parts.append(prompt)
    resp = client.models.generate_content(model=model, contents=parts)
    return parse_json_response((resp.text or "").strip())
