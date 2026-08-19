#!/usr/bin/env python3
"""Build blank-config hardness packs: k-field blanks (2–5) and component shells.

Protocol (same binary A/B as 1-blank):
  - Primary target attribute stays the scored A/B question.
  - Extra blanks remove context (other fields → <BLANK>).
  - ``component`` mode blanks every leaf in pose/move/path components
    (hold untouched) so the config reads like empty component shells.

Outputs, for each mode, original + counterbalanced JSON and a glossary_en
English/schema-YAML wrap (same prompt family as Tables 1–3).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_h4_blank_config_mcq import (  # noqa: E402
    BLANK,
    BASE,
    GT,
    EVAL,
    REPO,
    _blank_question,
    _option_rows,
    _reversed,
    _target_container,
)
from build_h4_glossary_english import (  # noqa: E402
    ANSWER,
    DESCRIPTIONS,
    SYSTEM,
    SCHEMA_YAML,
    _payload as _glossary_payload,
    embodiment_schema_block,
)

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
HOLD_TYPES = {"hold"}


def _stable_pick(key: str, items: list[Any], n: int) -> list[Any]:
    if n <= 0 or not items:
        return []
    scored = []
    for i, item in enumerate(items):
        digest = hashlib.sha256(f"{key}:{i}:{item}".encode()).hexdigest()
        scored.append((digest, i, item))
    scored.sort()
    return [item for _, _, item in scored[: min(n, len(items))]]


def _iter_blankable_leaves(
    movements: list[dict[str, Any]],
) -> list[tuple[int, str, dict[str, Any], str]]:
    """Return (component_index, component_type, container, key) for blankable leaves."""
    out: list[tuple[int, str, dict[str, Any], str]] = []
    for ci, movement in enumerate(movements):
        kind = str(movement.get("type", "")).lower()
        if kind in HOLD_TYPES or kind not in {"pose", "movement", "path", "move"}:
            continue
        parameters = movement.get("parameters") or {}
        nested = parameters.get(kind) if isinstance(parameters.get(kind), dict) else None
        if kind == "pose":
            pose = nested.get("pose") if nested and isinstance(nested.get("pose"), dict) else (
                nested if nested is not None else parameters
            )
            if not isinstance(pose, dict):
                continue
            for key, value in pose.items():
                if key in {"speed", "hold_time", "duration"}:
                    continue
                if value is None or value == "":
                    continue
                out.append((ci, "pose", pose, key))
        elif kind in {"movement", "move"}:
            mv = nested if nested is not None else parameters
            if not isinstance(mv, dict):
                continue
            for key in ("repetition", "rep", "speed", "hold_time"):
                if key in mv and mv[key] not in (None, ""):
                    out.append((ci, "movement", mv, key))
            joints = mv.get("joints")
            if isinstance(joints, list):
                for ji, joint in enumerate(joints):
                    if not isinstance(joint, dict):
                        continue
                    for key, value in joint.items():
                        if value in (None, ""):
                            continue
                        # tag joint leaves with synthetic key path for uniqueness
                        out.append((ci, "movement", joint, key))
        elif kind == "path":
            path = nested if nested is not None else parameters
            if not isinstance(path, dict):
                continue
            for key, value in path.items():
                if value in (None, ""):
                    continue
                out.append((ci, "path", path, key))
    return out


def _blank_leaves(
    movements: list[dict[str, Any]],
    leaves: list[tuple[int, str, dict[str, Any], str]],
) -> int:
    n = 0
    for _ci, _kind, container, key in leaves:
        if key in container and container[key] != BLANK:
            container[key] = BLANK
            n += 1
    return n


def _component_shell(movements: list[dict[str, Any]]) -> int:
    """Blank every leaf inside pose/move/path components. Returns blank count."""
    leaves = _iter_blankable_leaves(movements)
    return _blank_leaves(movements, leaves)


def _english_hardness_body(row: dict[str, Any], *, mode: str, k: int | None) -> str:
    if mode == "component":
        how = (
            "Every attribute inside pose / movement / path components has been "
            "replaced with a blank token (component shells only; hold kept)."
        )
    else:
        how = (
            f"Exactly {k} attribute fields in the full config have been replaced "
            f"with blank tokens."
        )
    return (
        f"The full Human GT motion config for gesture '{row['cue']}' on the "
        f"{row['platform']} embodiment is shown below. {how} "
        f"Recover the value of '{row['attribute']}' for the "
        f"{row['component_number']}th {row['component_type']} component.\n"
        "Use the gesture meaning, the embodiment schema, and the remaining "
        "config context to choose the exact attribute value from the two "
        "candidates.\n\n"
        "[FULL GT MOVEMENTS CONFIG]\n"
        + row["_blanked_movements_json"]
    )


def _wrap_glossary(row: dict[str, Any], *, mode: str, k: int | None, reversed_pair: bool) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out.pop("_blanked_movements_json", None)
    base_id = re.sub(r"\.reversed$", "", out["question_id"])
    suffix = f".blank_{mode}" if mode == "component" else f".blank_k{k}"
    if not base_id.endswith(suffix):
        # strip prior .blank / .glossary_en noise then append
        base_id = re.sub(r"\.blank(_k\d+|_component)?(\.glossary_en)?$", "", base_id)
        base_id = f"{base_id}{suffix}.glossary_en"
    if reversed_pair:
        out["question_id"] = f"{base_id}.reversed"
        out["paired_question_id"] = base_id
    else:
        out["question_id"] = base_id
        out.pop("paired_question_id", None)
    out["experiment_id"] = f"schema_yaml_en_{mode}" if mode == "component" else f"schema_yaml_en_k{k}"
    out["system_prompt"] = SYSTEM
    out["response_mode"] = "value"
    out["answer_instruction"] = ANSWER
    out["schema_source"] = str(SCHEMA_YAML.relative_to(REPO))
    # Rebuild English prompt from blanked JSON stored before wrap
    blanked = row["_blanked_movements_json"]
    out["_blanked_movements_json"] = blanked
    out["prompt"] = (
        f"[GESTURE MEANING]\n{DESCRIPTIONS[out['cue']]}\n\n"
        f"{embodiment_schema_block(str(out['platform']))}\n\n"
        f"{_english_hardness_body(out, mode=mode, k=k)}"
    )
    out.pop("_blanked_movements_json", None)
    if re.search(r"[가-힣]", out["prompt"]):
        raise RuntimeError(f"Korean remains in {out['question_id']}")
    return out


def build_hardness_question(
    base: dict[str, Any],
    platform_gt: dict[str, Any],
    *,
    mode: str,
    k: int | None,
) -> dict[str, Any]:
    """Build one hardness question. mode in {fields, component}."""
    # Start from the standard 1-blank question machinery for correct/foil/options.
    one = _blank_question(base, platform_gt)
    movements = copy.deepcopy(platform_gt["movements"])
    primary_ci = int(base["component_index"])
    primary_attr = str(base["attribute"])
    container, key = _target_container(movements[primary_ci], primary_attr)
    literal_correct = container[key]

    if mode == "component":
        n_blank = _component_shell(movements)
        # ensure primary is blank
        container, key = _target_container(movements[primary_ci], primary_attr)
        if container.get(key) != BLANK:
            container[key] = BLANK
            n_blank += 1
        input_format = "complete_gt_movements_component_shells"
        q_suffix = ".blank_component"
        rule = f"component_shells+{one['mutation_rule']}"
    else:
        assert k is not None and k >= 1
        # Blank primary first
        container[key] = BLANK
        leaves = _iter_blankable_leaves(movements)
        primary_leaf = (primary_ci, str(base["component_type"]).lower(), container, key)
        # Normalize component_type movement vs move
        others = []
        for leaf in leaves:
            ci, kind, cont, lk = leaf
            if ci == primary_ci and cont is container and lk == key:
                continue
            # also skip if already blank
            if cont.get(lk) == BLANK:
                continue
            others.append(leaf)
        extra = _stable_pick(one["question_id"] + f".k{k}", others, k - 1)
        n_blank = 1 + _blank_leaves(movements, extra)
        # If not enough other leaves, blank what we can (still record actual count)
        input_format = f"complete_gt_movements_with_{k}_blanks"
        q_suffix = f".blank_k{k}"
        rule = f"k{k}_context_ablation+{one['mutation_rule']}"

    blanked_json = json.dumps(movements, indent=2, ensure_ascii=False)
    actual_blanks = blanked_json.count(BLANK)
    if actual_blanks < 1:
        raise RuntimeError(f"{base['question_id']}: no blanks applied")

    prompt = (
        f"Gesture '{base['cue']}' / {base['platform']}: recover "
        f"'{primary_attr}' of the {base['component_number']}th "
        f"{base['component_type']} (A/B). Config has {actual_blanks} blank "
        f"token(s).\n\n[FULL GT MOVEMENTS CONFIG]\n{blanked_json}"
    )

    # literal foil same as 1-blank
    if primary_attr == "repetition":
        literal_foil = 1 if float(literal_correct) > 1 else 3
    else:
        literal_foil = base["foil_value"]

    return {
        **{kk: vv for kk, vv in one.items() if kk not in {"prompt", "input_format", "mutation_rule", "blank_target", "question_id"}},
        "question_id": f"{base['question_id']}{q_suffix}",
        "prompt": prompt,
        "options": _option_rows(literal_correct, literal_foil, base["correct_option"]),
        "correct_value": literal_correct,
        "foil_value": literal_foil,
        "mutation_rule": rule,
        "input_format": input_format,
        "blank_count": actual_blanks,
        "blank_mode": mode,
        "blank_target": {
            "component_index": primary_ci,
            "component_type": base["component_type"],
            "attribute": primary_attr,
            "field_key": key,
        },
        "_blanked_movements_json": blanked_json,
    }


def _write_pack(
    questions: list[dict[str, Any]],
    *,
    stem: str,
    mode: str,
    k: int | None,
) -> tuple[Path, Path, Path, Path]:
    # Strip helper field for raw blank packs
    clean = []
    for q in questions:
        row = {kk: vv for kk, vv in q.items() if not kk.startswith("_")}
        clean.append(row)

    summary = {
        "questions": len(clean),
        "blank_mode": mode,
        "blank_k": k,
        "by_platform": dict(Counter(row["platform"] for row in clean)),
        "by_attribute": dict(Counter(row["attribute"] for row in clean)),
        "blank_count_hist": dict(Counter(row.get("blank_count", 0) for row in clean)),
        "correct_options": dict(Counter(row["correct_option"] for row in clean)),
    }
    definition = {
        "format": (
            "complete GT movements with component shells blanked"
            if mode == "component"
            else f"complete GT movements with {k} blank fields (context ablation)"
        ),
        "source": str(GT.relative_to(REPO)),
        "primary_task": "binary A/B for one target attribute",
        "hardness": mode if mode == "component" else f"k={k}",
    }
    payload = {"definition": definition, "summary": summary, "questions": clean}
    rev = {
        **payload,
        "definition": {**definition, "option_order": f"exact A/B reversal of {stem}.json"},
        "questions": [_reversed(q) for q in clean],
        "summary": {
            **summary,
            "correct_options": dict(
                Counter(_reversed(q)["correct_option"] for q in clean)
            ),
        },
    }
    out = EVAL / f"{stem}.json"
    out_rev = EVAL / f"{stem}_counterbalanced.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_rev.write_text(json.dumps(rev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # glossary_en wrap (needs _blanked_movements_json on questions)
    g_rows = [_wrap_glossary(q, mode=mode, k=k, reversed_pair=False) for q in questions]
    g_rev = [_wrap_glossary(q, mode=mode, k=k, reversed_pair=True) for q in questions]
    # fix reversed options from rev pack
    for i, q in enumerate(rev["questions"]):
        g_rev[i]["options"] = q["options"]
        g_rev[i]["correct_option"] = q["correct_option"]

    g_stem = stem.replace("h4_gt_blank_", "h4_glossary_")
    g_out = EVAL / f"{g_stem}.json"
    g_out_rev = EVAL / f"{g_stem}_counterbalanced.json"
    g_out.write_text(
        json.dumps(_glossary_payload(g_rows, reversed_set=False), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    g_out_rev.write_text(
        json.dumps(_glossary_payload(g_rev, reversed_set=True), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(out)
    print(g_out)
    return out, out_rev, g_out, g_out_rev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        default="k2,k3,k4,k5,component",
        help="Comma list: k2,k3,k4,k5,component",
    )
    args = ap.parse_args()

    master = json.loads(GT.read_text(encoding="utf-8"))
    gt_by_key = {
        (int(entry["idx"]), platform): row
        for entry in master
        for platform, row in entry["platforms"].items()
        if platform in PLATFORMS
    }
    base_payload = json.loads(BASE.read_text(encoding="utf-8"))
    base_qs = [
        q
        for q in base_payload["questions"]
        if q["platform"] in PLATFORMS
    ]

    written: list[str] = []
    for token in [t.strip() for t in args.modes.split(",") if t.strip()]:
        if token == "component":
            mode, k, stem = "component", None, "h4_gt_blank_component"
        elif token.startswith("k") and token[1:].isdigit():
            k = int(token[1:])
            mode, stem = "fields", f"h4_gt_blank_k{k}"
        else:
            raise ValueError(f"unknown mode token: {token}")

        questions = [
            build_hardness_question(q, gt_by_key[(int(q["cue_idx"]), str(q["platform"]))], mode=mode, k=k)
            for q in base_qs
        ]
        _write_pack(questions, stem=stem, mode=mode, k=k)
        written.append(stem)

    print("wrote", written)


if __name__ == "__main__":
    main()
