"""Helpers for batched vLLM inference in verify/generation loops."""
from __future__ import annotations

from typing import Any

from vlm_client import VLMClient, vlm_batch_size


def vlm_generate_texts(
    vlm: VLMClient | None,
    backend: str,
    requests: list[dict[str, Any]],
) -> list[str]:
    """Run one or more VLM requests; batch when backend is vLLM local."""
    if not requests:
        return []
    if vlm is None:
        return [""] * len(requests)
    if len(requests) == 1 or vlm_batch_size(backend) <= 1:
        return [
            vlm.generate(
                r["prompt"],
                images=r.get("images"),
                videos=r.get("videos"),
            )
            for r in requests
        ]
    return vlm.generate_many(requests)
