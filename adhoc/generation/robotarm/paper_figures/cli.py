#!/usr/bin/env python3
"""Paper figure CLI for pilot-90 benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
for p in (_ROBOTARM.parents[2], _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot-90 paper figures")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("acc", help="Accuracy line plots (pose + movement)")
    sp.add_argument("--gemini-dir", default="data/results/verify/pilot90_gemini")
    sp.add_argument("--qwen32-dir", default="data/results/verify/pilot90_qwen32b")
    sp.add_argument("--qwen7-dir", default="data/results/verify/pilot90_qwen7b")
    sp.add_argument("--qwen3-dir", default="data/results/verify/pilot90_qwen3b")
    sp.add_argument("--out-dir", default=None)

    sp = sub.add_parser("qual", help="Qualitative grid (pose or movement)")
    sp.add_argument("--mode", choices=("pose", "movement"), required=True)
    sp.add_argument("--idx", default="all")
    sp.add_argument("--ncols", type=int, default=3)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("pairwise", help="8-panel pairwise figure")
    sp.add_argument("--idx", default="0,1,0,1")
    sp.add_argument("--pose-json", default=None)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("components", help="Pose / movement / path definition figure")
    sp.add_argument("--pose-idx", type=int, default=16)
    sp.add_argument("--movement-idx", type=int, default=59)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("persona", help="Persona variations (curated cue×persona matrix)")
    sp.add_argument("--matrix", default="v20", choices=("legacy", "v20"))
    sp.add_argument("--matrix-yml", default=None)
    sp.add_argument("--cue-idxs", default=None)
    sp.add_argument("--model", default="gemini-2.5-pro")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--skip-render", action="store_true")
    sp.add_argument("--force-render", action="store_true")
    sp.add_argument("--html-only", action="store_true")
    sp.add_argument("--out-dir", default=None)

    sp = sub.add_parser("persona-grid", help="10-persona subplot GIF per cue")
    sp.add_argument("--persona-dir", default=None)
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--ncol", type=int, default=5)
    sp.add_argument("--panel-size", type=int, default=220)

    sp = sub.add_parser("essence10", help="Essence-10 generate + HTML")
    sp.add_argument("--model", default="gemini-2.5-pro")
    sp.add_argument("--cues", default=None, help="Comma-separated essence cue names")
    sp.add_argument("--skip-generate", action="store_true")
    sp.add_argument("--skip-render", action="store_true")
    sp.add_argument("--no-open", action="store_true")

    args = p.parse_args()

    if args.cmd == "acc":
        from paper_figures.plot_acc import run

        run(
            gemini_dir=args.gemini_dir,
            qwen32_dir=args.qwen32_dir,
            qwen7_dir=args.qwen7_dir,
            qwen3_dir=args.qwen3_dir,
            out_dir=args.out_dir,
        )
    elif args.cmd == "qual":
        from paper_figures.qual_generation import build_grid

        build_grid(mode=args.mode, idx_arg=args.idx, ncols=args.ncols, out_path=Path(args.out) if args.out else None)
    elif args.cmd == "pairwise":
        from paper_figures.qual_pairwise import build

        build(idx_arg=args.idx, pose_json=args.pose_json, out=args.out)
    elif args.cmd == "components":
        from paper_figures.component_def import build

        build(pose_idx=args.pose_idx, movement_idx=args.movement_idx, out=args.out)
    elif args.cmd == "persona":
        from paper_figures.persona_change import run

        run(
            matrix=args.matrix,
            matrix_yml=args.matrix_yml,
            cue_idxs=args.cue_idxs,
            model=args.model,
            dry_run=args.dry_run,
            skip_render=args.skip_render,
            force_render=args.force_render,
            html_only=args.html_only,
            out_dir=args.out_dir,
        )
    elif args.cmd == "persona-grid":
        from paper_figures.persona_grid_gif import run

        run(
            persona_dir=args.persona_dir,
            out_dir=args.out_dir,
            ncol=args.ncol,
            panel_size=args.panel_size,
        )
    elif args.cmd == "essence10":
        from paper_figures.essence10 import run

        run(
            model=args.model,
            skip_generate=args.skip_generate,
            skip_render=args.skip_render,
            open_html=not args.no_open,
            cues_filter=args.cues,
        )


if __name__ == "__main__":
    main()
