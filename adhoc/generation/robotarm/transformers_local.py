"""In-process HuggingFace transformers inference for Qwen-VL (no vLLM, no HTTP)."""
from __future__ import annotations

import os
import tempfile
from typing import Any

from PIL import Image

_ENGINE: "TransformersLocalEngine | None" = None


class TransformersLocalEngine:
    """Load Qwen2.5-VL via transformers; works on older CUDA/driver stacks."""

    def __init__(self, *, model: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name = model or os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not visible. Run on a GPU node (salloc/sbatch), not a login node."
            )
        try:
            torch.zeros(1, device="cuda")
        except RuntimeError as e:
            raise RuntimeError(
                "PyTorch CUDA init failed (driver mismatch?). Run:\n"
                "  bash scripts/install_vlm_transformers.sh\n"
                f"Detail: {e}"
            ) from e

        n_gpu = torch.cuda.device_count()
        from hf_cache_setup import hub_model_cache_dir, resolve_model_load_path

        load_path, local_only, cache_msg = resolve_model_load_path(self.model_name)
        cache_dir = hub_model_cache_dir(self.model_name)
        print(
            f"[transformers] {self.model_name} | {n_gpu} GPU(s) | {cache_msg}",
            flush=True,
        )
        print(f"[transformers] load_path={load_path} local_files_only={local_only}", flush=True)
        print(f"[transformers] hub_cache={cache_dir}", flush=True)
        load_kw = {"trust_remote_code": True, "local_files_only": local_only}
        print("[transformers] loading processor…", flush=True)
        self.processor = AutoProcessor.from_pretrained(load_path, **load_kw)
        print(
            "[transformers] loading model weights to GPU (7B ~2–5 min, 32B ~10–20 min; "
            "no output until done)…",
            flush=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_path,
            dtype=torch.bfloat16,
            device_map="auto",
            **load_kw,
        )
        self.model.eval()
        print("[transformers] ready", flush=True)

    def generate(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        videos: list[str] | None = None,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        images = images or []
        videos = videos or []
        content: list[dict[str, Any]] = []
        tmp_paths: list[str] = []

        try:
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
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            from vlm_sampling import transformers_generate_kwargs

            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, **transformers_generate_kwargs())

            trimmed = [
                o[len(i) :]
                for i, o in zip(inputs.input_ids, out_ids)
            ]
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return (decoded[0] or "").strip()
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def get_transformers_engine(**kwargs: Any) -> TransformersLocalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TransformersLocalEngine(**kwargs)
    return _ENGINE
