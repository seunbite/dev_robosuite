#!/usr/bin/env python3
"""Build embodiment VLM cue-identification eval packs (yes/no + 2AFC AB/BA).

Sources (150 videos each = 30 cues × 5 embodiments):
  - Human GT renders
  - Agent grok45 short renders
  - Agent opus48 short renders
  - Gemini 2.5 Pro short renders

Tasks:
  1.1 yes/no — video + cue label (matched YES + mismatched NO)
  1.2 2AFC — true cue vs GT-motion-code foil, AB and BA orders

Pairwise foil policies (GT motion codes, shared across video sources):
  - easy: lowest attribute-F1 neighbor (type-seq mismatch / more edits preferred)
  - hard: highest attribute-F1 neighbor (type-seq match / fewer edits preferred)

Usage::

    python build_embodiment_cue_id_eval.py              # easy (default)
    python build_embodiment_cue_id_eval.py --foil hard  # hard-negative packs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_embodiment_h4 import score_instance  # noqa: E402

CUES = REPO / "data/seed/agent_fill/embodiment_short/cues.json"
GT_MASTER = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
)
OUT_ROOT = REPO / "data/results/evaluation/embodiment_cue_id"

# Stretch replaces quadruped for the short cue-ID suite (5 embodiments × 30 cues).
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
RENDER_ROOTS = {
    "gt": REPO / "data/results/render/embodiment_motion_code_gt",
    "agent_grok45": REPO / "data/results/render/embodiment_motion_code_agent_grok45",
    "agent_opus48": REPO / "data/results/render/embodiment_motion_code_agent_short_opus48",
    "agent_gemini25pro": REPO
    / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro",
}


def _stable_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _stable_rank(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _media_paths(render_root: Path, platform: str, idx: int, cue: str) -> dict[str, str | None]:
    stem = f"emb_g{idx:02d}_{cue}"
    if platform == "single_arm":
        gif = render_root / "manipulator" / "gif" / "IIWA" / f"{stem}.gif"
        mp4 = render_root / "manipulator" / "mp4" / "IIWA" / f"{stem}.mp4"
    else:
        gif = render_root / platform / "gif" / f"{stem}.gif"
        mp4 = render_root / platform / "mp4" / f"{stem}.mp4"
    # Prefer mp4 as the canonical video path (Babel renders are mp4). Even if the
    # local checkout only has gifs, write the mp4 relative path so cluster jobs
    # do not silently fall back to unsupported gif inputs.
    return {
        "gif": _rel(gif) if gif.is_file() else None,
        "mp4": _rel(mp4),
        "video": _rel(mp4),
    }


def _cue_text(entry: dict[str, Any]) -> str:
    return f"{entry['cue']}: {entry['description']}"


def _other_cue(cues: list[dict[str, Any]], idx: int, key: str) -> dict[str, Any]:
    others = [c for c in cues if int(c["idx"]) != idx]
    return others[_stable_index(key, len(others))]


def _foil_candidates(
    *,
    platform: str,
    tgt: dict[str, Any],
    cues: list[dict[str, Any]],
    by_idx: dict[int, dict[str, Any]],
    drop_zero_f1: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ti = int(tgt["idx"])
    ta = by_idx[ti]["platforms"][platform]
    cands: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for foil in cues:
        fi = int(foil["idx"])
        if fi == ti:
            continue
        fb = by_idx[fi]["platforms"][platform]
        scored = score_instance(ta, fb)
        attr = scored["attribute"]
        inst = scored["instance"]
        if inst["exact_match"]:
            continue
        meta = {
            "f1": float(attr["f1"]),
            "type_seq_match": bool(inst["component_type_sequence_exact"]),
            "n_edit": len(attr["missing"]) + len(attr["extra"]),
            "same_category": foil.get("category") == tgt.get("category"),
        }
        cands.append((foil, meta))
    if not cands:
        raise RuntimeError(f"no non-exact foil for {platform} g{ti:02d}")
    if drop_zero_f1:
        pos = [c for c in cands if c[1]["f1"] > 0]
        return pos or cands
    return cands


def _build_easy_foil_map(
    cues: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (platform, target_idx) -> farthest non-exact GT neighbor (easy neg)."""
    master = json.loads(GT_MASTER.read_text(encoding="utf-8"))
    by_idx = {int(e["idx"]): e for e in master}
    out: dict[tuple[str, int], dict[str, Any]] = {}

    for platform in PLATFORMS:
        for tgt in cues:
            ti = int(tgt["idx"])
            cands = _foil_candidates(
                platform=platform, tgt=tgt, cues=cues, by_idx=by_idx, drop_zero_f1=False
            )

            def rank_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple:
                foil, meta = item
                return (
                    -meta["f1"],
                    0 if meta["type_seq_match"] else 1,
                    meta["n_edit"],
                    0 if meta["same_category"] else 1,
                    -_stable_rank(f"{platform}.{ti:02d}->{int(foil['idx']):02d}"),
                )

            foil, meta = max(cands, key=rank_key)
            out[(platform, ti)] = {
                "foil": foil,
                "f1": round(meta["f1"], 4),
                "type_seq_match": meta["type_seq_match"],
                "n_edit": meta["n_edit"],
                "selection": "gt_motion_code_easy_negative",
            }
    return out


def _build_hard_foil_map(
    cues: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (platform, target_idx) -> nearest non-exact GT neighbor (hard neg)."""
    master = json.loads(GT_MASTER.read_text(encoding="utf-8"))
    by_idx = {int(e["idx"]): e for e in master}
    out: dict[tuple[str, int], dict[str, Any]] = {}

    for platform in PLATFORMS:
        for tgt in cues:
            ti = int(tgt["idx"])
            cands = _foil_candidates(
                platform=platform, tgt=tgt, cues=cues, by_idx=by_idx, drop_zero_f1=True
            )

            def rank_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple:
                foil, meta = item
                return (
                    meta["f1"],
                    1 if meta["type_seq_match"] else 0,
                    -meta["n_edit"],
                    1 if meta["same_category"] else 0,
                    -_stable_rank(f"hard.{platform}.{ti:02d}->{int(foil['idx']):02d}"),
                )

            foil, meta = max(cands, key=rank_key)
            out[(platform, ti)] = {
                "foil": foil,
                "f1": round(meta["f1"], 4),
                "type_seq_match": meta["type_seq_match"],
                "n_edit": meta["n_edit"],
                "selection": "gt_motion_code_hard_negative",
            }
    return out


def _foil_table(
    cues: list[dict[str, Any]],
    foils: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        platform: [
            {
                "target_idx": ti,
                "target": next(c["cue"] for c in cues if int(c["idx"]) == ti),
                "foil_idx": int(info["foil"]["idx"]),
                "foil": info["foil"]["cue"],
                "f1": info["f1"],
                "type_seq_match": info["type_seq_match"],
                "n_edit": info.get("n_edit"),
            }
            for (plat, ti), info in sorted(foils.items())
            if plat == platform
        ]
        for platform in PLATFORMS
    }


_EMBODIMENT_APPROX = (
    "Even if this robot's embodiment is limited for the cue, credit approximate "
    "realizations that use available motion to convey the same intent."
)


def _yesno_prompt(cue_line: str) -> str:
    return (
        "You are evaluating a short robot motion video.\n"
        f"{_EMBODIMENT_APPROX}\n"
        f"Candidate cue:\n- {cue_line}\n\n"
        "Does this video look like the candidate cue?\n"
        "Answer with exactly one token: YES or NO."
    )


def _pairwise_prompt(a_line: str, b_line: str) -> str:
    return (
        "You are evaluating a short robot motion video.\n"
        f"{_EMBODIMENT_APPROX}\n"
        "Which cue does this video look like?\n"
        f"A. {a_line}\n"
        f"B. {b_line}\n\n"
        "Answer with exactly one token: A or B."
    )


_FOIL_MODES = {
    "easy": {
        "builder": _build_easy_foil_map,
        "selection": "gt_motion_code_easy_negative",
        "map_name": "pairwise_easy_foil_map.json",
        "pair_tag": "pair",
        "pairwise_suffix": "pairwise",
        "cue_id_suffix": "cue_id",
        "summary_name": "summary.json",
        "desc": (
            "1.2 true cue vs GT-motion-code easy negative (lowest F1); "
            "AB and BA orders; same foil map for all video sources"
        ),
    },
    "hard": {
        "builder": _build_hard_foil_map,
        "selection": "gt_motion_code_hard_negative",
        "map_name": "pairwise_hard_foil_map.json",
        "pair_tag": "hardpair",
        "pairwise_suffix": "pairwise_hardneg",
        "cue_id_suffix": "cue_id_hardneg",
        "summary_name": "summary_hardneg.json",
        "desc": (
            "1.2 true cue vs GT-motion-code hard negative (highest F1 / nearest); "
            "AB and BA orders; same foil map for all video sources"
        ),
    },
}


def _build_for_source(
    source: str,
    render_root: Path,
    cues: list[dict[str, Any]],
    foils: dict[tuple[str, int], dict[str, Any]],
    *,
    foil_mode: str = "easy",
    include_yesno: bool = True,
) -> dict[str, Any]:
    mode = _FOIL_MODES[foil_mode]
    yesno: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    missing: list[str] = []
    pair_tag = mode["pair_tag"]

    for platform in PLATFORMS:
        for entry in cues:
            idx = int(entry["idx"])
            cue = str(entry["cue"])
            media = _media_paths(render_root, platform, idx, cue)
            if not media["video"]:
                missing.append(f"{source}/{platform}/g{idx:02d}_{cue}")
                continue

            base = {
                "source": source,
                "platform": platform,
                "cue_idx": idx,
                "cue": cue,
                "category": entry.get("category"),
                "description": entry.get("description", ""),
                "gif": media["gif"],
                "mp4": media["mp4"],
                "video": media["video"],
            }

            if include_yesno:
                yesno.append(
                    {
                        **base,
                        "question_id": f"{source}.{platform}.g{idx:02d}.yesno.match",
                        "task": "yesno",
                        "trial": "match",
                        "queried_cue": cue,
                        "queried_cue_idx": idx,
                        "queried_description": entry.get("description", ""),
                        "prompt": _yesno_prompt(_cue_text(entry)),
                        "correct_answer": "YES",
                    }
                )
                foil = _other_cue(cues, idx, f"{source}.{platform}.g{idx:02d}.yesno.foil")
                yesno.append(
                    {
                        **base,
                        "question_id": f"{source}.{platform}.g{idx:02d}.yesno.mismatch",
                        "task": "yesno",
                        "trial": "mismatch",
                        "queried_cue": foil["cue"],
                        "queried_cue_idx": int(foil["idx"]),
                        "queried_description": foil.get("description", ""),
                        "prompt": _yesno_prompt(_cue_text(foil)),
                        "correct_answer": "NO",
                    }
                )

            foil_info = foils[(platform, idx)]
            foil = foil_info["foil"]
            true_line = _cue_text(entry)
            foil_line = _cue_text(foil)
            for order in ("AB", "BA"):
                if order == "AB":
                    a_line, b_line = true_line, foil_line
                    correct = "A"
                else:
                    a_line, b_line = foil_line, true_line
                    correct = "B"
                pairwise.append(
                    {
                        **base,
                        "question_id": f"{source}.{platform}.g{idx:02d}.{pair_tag}.{order}",
                        "task": "pairwise",
                        "option_order": order,
                        "option_a_cue": entry["cue"] if order == "AB" else foil["cue"],
                        "option_b_cue": foil["cue"] if order == "AB" else entry["cue"],
                        "foil_cue": foil["cue"],
                        "foil_cue_idx": int(foil["idx"]),
                        "foil_selection": foil_info["selection"],
                        "foil_gt_f1": foil_info["f1"],
                        "foil_type_seq_match": foil_info["type_seq_match"],
                        "prompt": _pairwise_prompt(a_line, b_line),
                        "correct_answer": correct,
                    }
                )

    return {
        "definition": {
            "source": source,
            "render_root": _rel(render_root),
            "tasks": {
                "yesno": "1.1 video + cue label; match→YES, mismatch→NO (hashed foil)",
                "pairwise": mode["desc"],
            },
            "pairwise_foil_policy": mode["selection"],
            "platforms": list(PLATFORMS),
            "n_cues": len(cues),
        },
        "summary": {
            "videos_expected": len(PLATFORMS) * len(cues),
            "videos_missing": len(missing),
            "missing": missing,
            "yesno_questions": len(yesno),
            "pairwise_questions": len(pairwise),
        },
        "yesno": yesno,
        "pairwise": pairwise,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--foil",
        choices=sorted(_FOIL_MODES),
        default="easy",
        help="easy = farthest GT neighbor; hard = nearest GT neighbor",
    )
    ap.add_argument(
        "--sources",
        default="",
        help="Comma filter of video sources (default: all). e.g. gt,agent_gemini25pro",
    )
    ap.add_argument(
        "--pairwise-only",
        action="store_true",
        help="Skip rewriting yes/no packs (always for --foil hard)",
    )
    args = ap.parse_args()
    mode = _FOIL_MODES[args.foil]
    # Hard-neg shares yes/no with easy; only rewrite pairwise artifacts.
    pairwise_only = bool(args.pairwise_only or args.foil == "hard")

    cues = sorted(json.loads(CUES.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    foils = mode["builder"](cues)
    foil_path = OUT_ROOT / mode["map_name"]
    foil_path.write_text(
        json.dumps(
            {"selection": mode["selection"], "platforms": _foil_table(cues, foils)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(foil_path)

    want = {s.strip() for s in args.sources.split(",") if s.strip()} or set(RENDER_ROOTS)
    combined_summary: dict[str, Any] = {}
    for source, root in RENDER_ROOTS.items():
        if source not in want:
            continue
        payload = _build_for_source(
            source,
            root,
            cues,
            foils,
            foil_mode=args.foil,
            include_yesno=not pairwise_only,
        )
        out = OUT_ROOT / f"{source}_{mode['cue_id_suffix']}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        combined_summary[source] = payload["summary"]
        print(source, json.dumps(payload["summary"], ensure_ascii=False))
        print(out)

        tasks = ("pairwise",) if pairwise_only else ("yesno", "pairwise")
        for task in tasks:
            flat = {
                "definition": payload["definition"],
                "task": task,
                "summary": {"questions": len(payload[task])},
                "questions": payload[task],
            }
            suffix = mode["pairwise_suffix"] if task == "pairwise" else "yesno"
            # Keep legacy easy yes/no / pairwise names; hard uses *_hardneg.
            if args.foil == "easy" and task == "pairwise":
                path = OUT_ROOT / f"{source}_pairwise.json"
            elif args.foil == "easy" and task == "yesno":
                path = OUT_ROOT / f"{source}_yesno.json"
            else:
                path = OUT_ROOT / f"{source}_{suffix}.json"
            path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
            print(path)

    summary_path = OUT_ROOT / mode["summary_name"]
    summary_path.write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
