#!/usr/bin/env python3
"""Check HuggingFace hub cache completeness for a model id."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "adhoc" / "generation" / "robotarm"))

from hf_cache_setup import cache_is_complete, cache_shard_status, hub_model_cache_dir, setup_hf_cache  # noqa: E402


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-VL-32B-Instruct"
    setup_hf_cache()
    cache_dir = hub_model_cache_dir(model_id)
    present, expected, snap = cache_shard_status(model_id)
    complete = cache_is_complete(model_id)

    print(f"model:     {model_id}")
    print(f"cache_dir: {cache_dir}")
    print(f"snapshot:  {snap}")
    print(f"shards:    {present}/{expected} complete={complete}")

    if snap and expected:
        missing = []
        from hf_cache_setup import _expected_weight_files

        for name in _expected_weight_files(snap):
            p = snap / name
            if not p.is_file():
                missing.append(name)
            else:
                print(f"  ok  {name} ({p.stat().st_size // (1024**2)} MiB)")
        for name in missing:
            print(f"  MISS {name}")

    if complete:
        print("\nOK — load with snapshot path, no download needed.")
        print(f"  export HF_HUB_OFFLINE=1")
    elif present > 0:
        print("\nINCOMPLETE — HF will download ONLY missing shards (may look like re-download).")
        print("  Let it finish once, or delete cache dir and re-download cleanly.")


if __name__ == "__main__":
    main()
