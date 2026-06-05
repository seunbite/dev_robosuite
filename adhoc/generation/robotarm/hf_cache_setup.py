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


def hub_model_cache_dir(model_id: str, hf_home: Path | None = None) -> Path:
    """Path like .../hub/models--Qwen--Qwen2.5-VL-32B-Instruct"""
    if hf_home is not None:
        hub = Path(hf_home)
        if hub.name != "hub":
            hub = hub / "hub"
    elif os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    else:
        hub = setup_hf_cache() / "hub"
    return hub / ("models--" + model_id.replace("/", "--"))


def model_is_locally_cached(model_id: str) -> bool:
    """True if hub cache has at least one snapshot with config.json."""
    cache_dir = hub_model_cache_dir(model_id)
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and (snap / "config.json").is_file():
            return True
    return False


def prefer_local_files(model_id: str) -> bool:
    """Use cache-only load when offline flag set or weights already on disk."""
    if os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return True
    return model_is_locally_cached(model_id)
