"""Shared LLM sampling defaults for pilot-90 generation (env-overridable)."""
from __future__ import annotations

import os
from typing import Any


def max_new_tokens() -> int:
    return int(os.getenv("VLM_MAX_NEW_TOKENS", "2048"))


def sampling_temperature() -> float:
    return float(os.getenv("VLM_TEMPERATURE", "0.7"))


def transformers_generate_kwargs() -> dict[str, Any]:
    """Kwargs for HuggingFace model.generate (text-only or multimodal)."""
    cap = max_new_tokens()
    temp = sampling_temperature()
    if temp <= 0:
        return {"max_new_tokens": cap, "do_sample": False}
    kw: dict[str, Any] = {
        "max_new_tokens": cap,
        "do_sample": True,
        "temperature": temp,
    }
    top_p = os.getenv("VLM_TOP_P")
    if top_p is not None:
        kw["top_p"] = float(top_p)
    return kw


def vllm_sampling_params():
    """vLLM SamplingParams with same env defaults."""
    from vllm import SamplingParams

    cap = max_new_tokens()
    temp = sampling_temperature()
    kw: dict[str, Any] = {"max_tokens": cap}
    if temp <= 0:
        kw["temperature"] = 0.0
    else:
        kw["temperature"] = temp
        top_p = os.getenv("VLM_TOP_P")
        if top_p is not None:
            kw["top_p"] = float(top_p)
    return SamplingParams(**kw)
