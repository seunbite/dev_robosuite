"""Point HuggingFace caches to /data (avoid home disk quota on clusters)."""
from __future__ import annotations

import json
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


def cache_snapshot_dir(model_id: str) -> Path | None:
    """Newest snapshot dir with config.json, or None."""
    snapshots = hub_model_cache_dir(model_id) / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [
        p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _expected_weight_files(snapshot: Path) -> list[str]:
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8"))
        weight_map = meta.get("weight_map") or {}
        return sorted(set(weight_map.values()))
    singles = sorted(p.name for p in snapshot.glob("*.safetensors"))
    return singles


def cache_shard_status(model_id: str) -> tuple[int, int, Path | None]:
    """Return (present_shards, expected_shards, snapshot_dir)."""
    snap = cache_snapshot_dir(model_id)
    if snap is None:
        return 0, 0, None
    expected = _expected_weight_files(snap)
    if not expected:
        return 0, 0, snap
    present = sum(1 for name in expected if (snap / name).is_file() and (snap / name).stat().st_size > 0)
    return present, len(expected), snap


def cache_is_complete(model_id: str) -> bool:
    present, expected, _ = cache_shard_status(model_id)
    return expected > 0 and present == expected


def resolve_model_load_path(model_id: str) -> tuple[str, bool, str]:
    """
    Returns (load_path, local_files_only, status_message).

    Complete cache → load straight from snapshot dir (no hub network).
    Incomplete → hub id + network to fetch missing shards only.
    """
    if os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        snap = cache_snapshot_dir(model_id)
        if snap is None or not cache_is_complete(model_id):
            raise FileNotFoundError(
                f"HF_HUB_OFFLINE=1 but cache incomplete for {model_id}. "
                f"Run: python scripts/check_hf_model_cache.py {model_id}"
            )
        return str(snap), True, f"offline snapshot ({snap.name})"

    present, expected, snap = cache_shard_status(model_id)
    if snap is not None and cache_is_complete(model_id):
        return str(snap), True, f"complete cache snapshot {present}/{expected} shards"

    if snap is not None and expected > 0:
        return (
            model_id,
            False,
            f"INCOMPLETE cache {present}/{expected} shards at {snap} — downloading missing files only",
        )

    return model_id, False, "not in cache — full download"


def prefer_local_files(model_id: str) -> bool:
    """Backward compat."""
    _, local_only, _ = resolve_model_load_path(model_id)
    return local_only
