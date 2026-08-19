#!/usr/bin/env python3
"""Re-render tiago exp1 originals + correction rounds (no VLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    config_snapshot_path,
    correction_cue_dir,
    default_render_dir,
    list_correction_rounds,
    load_config_row,
    render_instance,
)
from pilot40_paths import result_config_path  # noqa: E402


def _cue_list_from_html(
    *,
    cues: str | None,
    model_tag: str,
    exp_id: str,
) -> list[str]:
    if cues and cues.strip().lower() not in ("auto", "none", ""):
        return [c.strip() for c in cues.split(",") if c.strip()]
    from build_correction_evolution_html import discover_corrected_cues  # noqa: WPS433

    discovered = discover_corrected_cues(model_tag=model_tag, exp_id=exp_id)
    if not discovered:
        raise SystemExit("No correction cues found; pass --cues explicitly.")
    return [c for _, c in discovered]


def rerender(
    cues: str | None = "auto",
    model_tag: str = "gemini-2.5-pro",
    exp_id: str = "1",
    include_original: bool = True,
    include_corrections: bool = True,
) -> None:
    """Re-render motion GIFs for cues in correction_evolution.html (no VLM)."""
    if isinstance(cues, (list, tuple)):
        cues = ",".join(str(c) for c in cues)
    cue_list = _cue_list_from_html(cues=cues, model_tag=model_tag, exp_id=exp_id)
    print(f"[cues] {', '.join(cue_list)} ({len(cue_list)} total)")
    cfg_path = result_config_path(exp_id, model_tag)
    exp_render_dir = _REPO / "data/results/render/tiago" / f"exp{exp_id}_{model_tag}"

    if include_original:
        for cue in cue_list:
            row = load_config_row(cfg_path, cue=cue)
            print(f"[original] {cue} → {exp_render_dir}")
            render_instance(
                "tiago",
                row,
                config_path=cfg_path,
                render_dir=exp_render_dir,
                model_tag=model_tag,
                exp_id=exp_id,  # type: ignore[arg-type]
                round_n=1,
                ignore_xyz=False,
                topk=None,
            )

    if include_corrections:
        for cue in cue_list:
            cue_dir = correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
            rounds = list_correction_rounds(cue_dir)
            for rn in rounds:
                snap = config_snapshot_path("tiago", exp_id, model_tag, cue, rn)  # type: ignore[arg-type]
                if not snap.is_file():
                    print(f"[skip] missing config: {snap}")
                    continue
                rows = json.loads(snap.read_text(encoding="utf-8"))
                row = rows[0] if isinstance(rows, list) else rows
                out = default_render_dir("tiago", model_tag, cue, exp_id, rn)  # type: ignore[arg-type]
                print(f"[corr{rn}] {cue} → {out}")
                render_instance(
                    "tiago",
                    row,
                    config_path=snap,
                    render_dir=out,
                    model_tag=model_tag,
                    exp_id=exp_id,  # type: ignore[arg-type]
                    round_n=rn,
                    ignore_xyz=True,
                    topk=None,
                )

    from build_correction_evolution_html import build_evolution_html  # noqa: WPS433

    html = build_evolution_html(model_tag=model_tag, cues=cue_list)
    print(f"[evolution html] {html}")


if __name__ == "__main__":
    fire.Fire(rerender)
