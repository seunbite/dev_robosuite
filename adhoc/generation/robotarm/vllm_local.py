"""In-process vLLM inference for Qwen-VL (no HTTP server)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

# vLLM v1 pulls flashinfer → torch_c_dlpack_ext, which must match the installed
# torch wheel (common failure on Babel: torch 2.4 + torch26 dlpack .so).
# Legacy engine avoids that import chain unless the user opts in.
if os.getenv("VLLM_USE_V1") is None:
    os.environ["VLLM_USE_V1"] = "0"

_ENGINE: "VLLMLocalEngine | None" = None


class VLLMLocalEngine:
    """Load Qwen-VL once; generate from text + PIL images."""

    def __init__(
        self,
        *,
        model: str | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        self.model = model or os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        self.tensor_parallel_size = int(
            tensor_parallel_size or os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
        )
        self.max_model_len = int(max_model_len or os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        self.gpu_memory_utilization = float(
            gpu_memory_utilization or os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
        )
        self.enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "10"))

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        print(
            f"[vllm-local] loading {self.model} "
            f"tp={self.tensor_parallel_size} max_len={self.max_model_len} "
            f"eager={self.enforce_eager} max_num_seqs={self.max_num_seqs}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model)
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={"image": 4},
            trust_remote_code=True,
        )
        from vlm_sampling import vllm_sampling_params

        self._sampling_factory = vllm_sampling_params
        print("[vllm-local] ready", flush=True)

    def _prepare_request(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        videos: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        from qwen_vl_utils import process_vision_info

        images = images or []
        videos = videos or []
        content: list[dict[str, Any]] = []
        tmp_paths: list[str] = []

        for img in images:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img.convert("RGB").save(path)
            tmp_paths.append(path)
            content.append({"type": "image", "image": path})
        for vpath in videos:
            content.append({"type": "video", "video": vpath})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {"prompt": prompt_text, "multi_modal_data": mm_data}, tmp_paths

    def generate(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        videos: list[str] | None = None,
    ) -> str:
        req, tmp_paths = self._prepare_request(prompt, images, videos)
        try:
            outputs = self.llm.generate(
                req,
                sampling_params=self._sampling_factory(),
            )
            return (outputs[0].outputs[0].text or "").strip()
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[str]:
        """Batch inference. Each item: {prompt, images?, videos?}."""
        if not requests:
            return []
        if len(requests) == 1:
            req = requests[0]
            return [
                self.generate(
                    req["prompt"],
                    images=req.get("images"),
                    videos=req.get("videos"),
                )
            ]

        inputs: list[dict[str, Any]] = []
        all_tmp: list[str] = []
        try:
            for req in requests:
                inp, tmp_paths = self._prepare_request(
                    req["prompt"],
                    images=req.get("images"),
                    videos=req.get("videos"),
                )
                inputs.append(inp)
                all_tmp.extend(tmp_paths)

            outputs = self.llm.generate(
                inputs,
                sampling_params=self._sampling_factory(),
            )
            return [(o.outputs[0].text or "").strip() for o in outputs]
        finally:
            for p in all_tmp:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def get_vllm_engine(**kwargs: Any) -> VLLMLocalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VLLMLocalEngine(**kwargs)
    return _ENGINE
