"""
VLM clients for humeaneval-style runs: Gemini (google.genai) and Anthropic (optional).
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any


def _mime_for_path(p: Path) -> str:
    m, _ = mimetypes.guess_type(str(p))
    if m:
        return m
    s = p.suffix.lower()
    if s == ".gif":
        return "image/gif"
    if s in (".png",):
        return "image/png"
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def vlm_multimodal_call(
    model_id: str,
    user_text: str,
    media: list[tuple[Path, str | None]],
) -> str:
    """
    One round: text + N files. ``media`` is (path, mime or None to guess).

    ``model_id``:
      - contains ``gemini`` / starts with model name for ``google.genai``:
        e.g. ``gemini-2.5-flash`` (uses env ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``)
      - ``claude-...`` (uses env ``ANTHROPIC_API_KEY``)
    """
    mid = model_id.strip()
    mlow = mid.lower()
    if mlow.startswith("claude"):
        return _call_anthropic(mid, user_text, media)
    return _call_gemini(mid, user_text, media)


def _call_gemini(model_id: str, user_text: str, media: list[tuple[Path, str | None]]) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    client = genai.Client(api_key=api_key)
    parts: list[Any] = []
    for p, mt in media:
        data = Path(p).read_bytes()
        mime = (mt or _mime_for_path(p)).split(";")[0].strip()
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(user_text)
    resp = client.models.generate_content(model=model_id, contents=parts)
    return (resp.text or "").strip()


def _call_anthropic(model_id: str, user_text: str, media: list[tuple[Path, str | None]]) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install `anthropic` for Claude, or use a Gemini model_id.") from e
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY for Claude.")
    client = anthropic.Anthropic(api_key=key)
    content: list[dict[str, Any]] = []
    for p, mt in media:
        path = Path(p)
        mime = (mt or _mime_for_path(p)).split(";")[0].strip()
        if mime.startswith("video/"):
            import base64

            b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "video", "source": {"type": "base64", "media_type": mime, "data": b64}})
        else:
            import base64

            b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
            if not mime.startswith("image/"):
                mime = "image/png"
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )
    content.append({"type": "text", "text": user_text})
    msg = client.messages.create(model=model_id, max_tokens=4096, messages=[{"role": "user", "content": content}])
    parts = [b for b in (msg.content or []) if getattr(b, "type", None) == "text"]
    if not parts:
        return ""
    return (parts[0].text or "").strip()
