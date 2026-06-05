"""Point HuggingFace caches to /data (avoid home disk quota on clusters)."""
from __future__ import annotations

import os
from pathlib import Path


def setup_hf_cache(hf_home: str | Path | None = None) -> Path:
    """
    Set HF_HOME / HUGGINGFACE_HUB_CACHE before any model download.

    Priority: explicit arg > HF_HOME env > /data/user_data/$USER/hf_cache
    """
    if hf_home is None:
        hf_home = os.environ.get("HF_HOME")
    if hf_home is None:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
        hf_home = f"/data/user_data/{user}/hf_cache"

    root = Path(hf_home).expanduser().resolve()
    hub = root / "hub"
    transformers = root / "transformers"
    datasets = root / "datasets"
    for d in (hub, transformers, datasets):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    os.environ["HF_DATASETS_CACHE"] = str(datasets)
    return root
