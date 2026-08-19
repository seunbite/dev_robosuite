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
    sp.add_argument("--good-idx", default=None)
    sp.add_argument("--bad-idx", default=None)
    sp.add_argument("--ncols", type=int, default=3)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("pairwise", help="8-panel pairwise figure")
    sp.add_argument("--idx", default="0,1,0,1")
    sp.add_argument("--pose-json", default=None)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("qual-verify", help="Verification qual grid (task4 pose / task10 motion)")
    sp.add_argument("--mode", choices=("pose", "motion"), required=True)
    sp.add_argument("--idx", default="all")
    sp.add_argument("--good-idx", default=None)
    sp.add_argument("--bad-idx", default=None)
    sp.add_argument("--pose-json", default=None)
    sp.add_argument("--ncols", type=int, default=4)
    sp.add_argument("--cell-size", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("components", help="Pose / movement / path definition figure")
    sp.add_argument("--embodiment", choices=("IIWA", "Panda", "Tiago", "Quadruped"), default="IIWA")
    sp.add_argument("--all-embodiments", action="store_true")
    sp.add_argument("--pose-idx", type=int, default=None)
    sp.add_argument("--path-variant", choices=("y", "x-pos"), default="y")
    sp.add_argument("--all-path-variants", action="store_true")
    sp.add_argument("--force", action="store_true")
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

    sp = sub.add_parser("context", help="Exp12 context variation (persona + scenario)")
    sp.add_argument("--catalog-yml", default=None)
    sp.add_argument("--model", default="gemini-2.5-pro")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--skip-render", action="store_true")
    sp.add_argument("--force-render", action="store_true")
    sp.add_argument("--html-only", action="store_true")
    sp.add_argument("--out-dir", default=None)
    sp.add_argument("--open", action="store_true")

    sp = sub.add_parser("exp11-input", help="Exp11 repetition pairwise input HTML")
    sp.add_argument("--open", action="store_true")

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

        build_grid(
            mode=args.mode,
            idx_arg=args.idx,
            ncols=args.ncols,
            out_path=Path(args.out) if args.out else None,
            good_idxs=[int(x) for x in args.good_idx.split(",")] if args.good_idx else None,
            bad_idxs=[int(x) for x in args.bad_idx.split(",")] if args.bad_idx else None,
        )
    elif args.cmd == "pairwise":
        from paper_figures.qual_pairwise import build

        build(idx_arg=args.idx, pose_json=args.pose_json, out=args.out)
    elif args.cmd == "qual-verify":
        from paper_figures.qual_verification import build_grid

        from paper_figures.qual_generation import VERIFY_CELL_SIZE

        build_grid(
            mode=args.mode,
            idx_arg=args.idx,
            ncols=args.ncols,
            good_idxs=[int(x) for x in args.good_idx.split(",")] if args.good_idx else None,
            bad_idxs=[int(x) for x in args.bad_idx.split(",")] if args.bad_idx else None,
            pose_json=Path(args.pose_json) if args.pose_json else None,
            out_path=Path(args.out) if args.out else None,
            cell_size=args.cell_size or VERIFY_CELL_SIZE,
            seed=args.seed,
        )
    elif args.cmd == "components":
        from paper_figures.component_def import build

        if args.all_embodiments:
            from paper_figures.component_def import EMBODIMENTS, build as build_one

            for emb in EMBODIMENTS:
                build_one(
                    embodiment=emb,
                    pose_idx=args.pose_idx,
                    path_variant=args.path_variant,
                    force=args.force,
                )
        else:
            build(
                embodiment=args.embodiment,
                pose_idx=args.pose_idx,
                path_variant=args.path_variant,
                out=args.out,
                force=args.force,
            )
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
    elif args.cmd == "context":
        from exp12_context_variation import build_input_html, main as exp12_main

        if args.html_only and not args.force_render:
            out = build_input_html(
                catalog_yml=Path(args.catalog_yml) if args.catalog_yml else None,
            )
            if args.open:
                import subprocess

                subprocess.run(["open", str(out)], check=False)
        else:
            import sys

            argv = ["exp12_context_variation.py"]
            if args.catalog_yml:
                argv += ["--catalog-yml", args.catalog_yml]
            if args.model:
                argv += ["--model", args.model]
            if args.dry_run:
                argv.append("--dry-run")
            if args.skip_render:
                argv.append("--skip-render")
            if args.force_render:
                argv.append("--force-render")
            if args.html_only:
                argv.append("--html-only")
            if args.out_dir:
                argv += ["--out-dir", args.out_dir]
            if args.open:
                argv.append("--open")
            sys.argv = argv
            exp12_main()
    elif args.cmd == "exp11-input":
        from exp11_rep_pairwise import main as exp11_main

        import sys

        sys.argv = ["exp11_rep_pairwise.py"] + (["--open"] if args.open else [])
        exp11_main()
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
