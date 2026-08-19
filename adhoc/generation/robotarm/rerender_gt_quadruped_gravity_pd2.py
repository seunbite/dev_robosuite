#!/usr/bin/env python3
"""Re-render Human-GT quadruped (30) with A_gravity_pd2; refresh emb_*.gif/mp4."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "adhoc" / "generation" / "robotarm"))

from motion_code_to_config import render_quadruped  # noqa: E402
from motion_media_paths import gif_to_mp4  # noqa: E402


def main() -> None:
    cfg_path = _REPO / "data/results/motion_configs/embodiment_motion_code_gt/result_quadruped.json"
    out_root = _REPO / "data/results/render/embodiment_motion_code_gt"
    rows = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(
        f"[{time.strftime('%H:%M:%S')}] start GT quadruped re-render n={len(rows)} force=True",
        flush=True,
    )
    print(
        "recipe: earth-gravity + no-hold-base + pose-mode gravity + pd-scale 2.0",
        flush=True,
    )
    t0 = time.time()
    results = render_quadruped(rows, config_path=cfg_path, out_root=out_root, force=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] render done ok={len(results)}/{len(rows)} "
        f"elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    gif_dir = out_root / "quadruped" / "gif"
    n_mp4 = 0
    for row in rows:
        idx, cue = int(row["idx"]), str(row["cue"])
        emb = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        if not emb.is_file():
            print(f"MISSING emb {emb.name}", flush=True)
            continue
        mp4 = emb.with_suffix(".mp4")
        try:
            gif_to_mp4(emb, mp4)
            n_mp4 += 1
            print(f"mp4 {mp4.name} ({mp4.stat().st_size // 1024}KB)", flush=True)
        except Exception as e:
            print(f"mp4 FAIL {emb.name}: {e}", flush=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] done emb_gif={len(results)} emb_mp4={n_mp4}",
        flush=True,
    )


if __name__ == "__main__":
    main()
