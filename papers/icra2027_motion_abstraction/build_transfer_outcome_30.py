#!/usr/bin/env python3
"""Build a provisional 30-cue transfer-outcome table for human and LLM SMPs.

The common semantic source is the human-authored G1 program.  Human-authored
programs on each target morphology are coded relative to that source.  LLM
programs are first checked against the target morphology's human-authored O/L/M
realization, using only attributes marked recognition-critical in the cue
inventory.  A failed critical check receives L/D.  Otherwise it receives the
same transfer analysis as any other target program.

Primary outcomes are P (preserved), R (rematerialized), and L/D (lost or
distorted).  Augmentation is an orthogonal +A flag because an added cue can
coexist with any primary outcome.  This is program-derived coding and should be
validated by independent human coders and recognition judgments.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "motion_code"))

from classify_rematerialization import (  # noqa: E402
    classify,
    dyn_keys,
    stroke_match,
)
from score_nonhumanoid_olm import (  # noqa: E402
    _code,
    load_by_cue,
    parse_moves,
    parse_paths,
    score_location,
    score_movement,
    score_orientation,
)

GT_PATH = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
LLM_PATH = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
CRITICAL_PATH = HERE / "gesture_critical_factors_90.md"
DIFFICULTY_PATH = HERE / "embodiment_physical_difficulty_30.json"
OUT_MD = HERE / "transfer_outcomes_30.md"
OUT_CSV = HERE / "transfer_outcomes_30.csv"
OUT_JSON = HERE / "transfer_outcomes_30.json"
OUT_TEX = HERE / "transfer_outcomes_30.tex"

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
PLATFORM_LABEL = {
    "humanoid": "G1",
    "tiago": "TIAGo",
    "single_arm": "IIWA",
    "stretch": "Stretch",
    "roomba": "Roomba",
}
DIFFICULTY_KEY = {"single_arm": "Single Arm", "stretch": "Stretch", "roomba": "Roomba"}


def load_critical() -> dict[str, dict[str, bool | str]]:
    rows: dict[str, dict[str, bool | str]] = {}
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    )
    for line in CRITICAL_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        _, cue, location, orientation, movement = match.groups()
        rows[cue] = {
            "L": location.strip() != "None",
            "O": orientation.strip() != "None",
            "M": movement.strip() != "None",
            "location": location.strip(),
            "orientation": orientation.strip(),
            "movement": movement.strip(),
        }
    return rows


def carrier_families(code: str) -> set[str]:
    families: set[str] = set()
    for move in parse_moves(code):
        joint = str(move.get("joint") or "").lower()
        if any(token in joint for token in ("head", "camera", "neck")):
            families.add("head")
        elif any(token in joint for token in ("torso", "spine", "body")):
            families.add("torso")
        elif "base" in joint:
            families.add("base")
        else:
            families.add("upper_limb")
    for path in parse_paths(code):
        families.add("base" if path.get("carrier") == "base" else "upper_limb")
    return families


def _oscillates(code: str) -> bool:
    return any(sign == "pm" for _, sign in dyn_keys(code))


def _has_event(code: str) -> bool:
    """Whether the program contains an articulated or Cartesian/base event."""
    return bool(parse_moves(code) or parse_paths(code))


def _is_spatial_envelope(code: str) -> bool:
    """A visible traced envelope; only tiny base arcs count as in-place yaw."""
    paths = parse_paths(code)
    for path in paths:
        if path.get("shape") != "arc":
            continue
        radius = path.get("r")
        if path.get("carrier") != "base" and (radius is None or radius > 0):
            return True
        if path.get("carrier") == "base" and (radius or 0.0) >= 0.08:
            return True
    lines = [path for path in paths if path.get("shape") == "line"]
    return len({(path.get("axis"), path.get("sign")) for path in lines}) >= 3


def base_outcome(
    source: str,
    target: str,
    platform: str,
    critical: dict,
    physically_difficult: bool,
) -> tuple[str, bool, str]:
    """Return primary outcome, augmentation flag, and an auditable reason."""
    # Source events are transfer-defining only when Movement is marked
    # recognition-critical; approach/retraction used to stage a static emblem
    # does not turn that emblem into a dynamic cue.
    source_dynamic = bool(critical["M"] and _has_event(source))
    target_dynamic = _has_event(target)
    strategy = classify(source, target)
    source_carriers = carrier_families(source)
    target_carriers = carrier_families(target)
    same_carrier = bool(source_carriers & target_carriers)
    augmented = strategy == "additive" or (not critical["M"] and target_dynamic)

    if platform == "humanoid" and source == target:
        return "P", False, "common human-authored source"

    if not source_dynamic:
        if platform == "roomba":
            return "L/D", augmented, "static hand/body evidence unavailable on a planar base"
        if physically_difficult:
            return "L/D", augmented, "indispensable carrier structure is unavailable"
        return "P", augmented, "static evidence retained on an analogous articulated carrier"

    if not target_dynamic:
        return "L/D", augmented, "required event structure is absent"

    # A complete traced envelope is a direct preservation when the carrier is
    # analogous and a rematerialization when it moves to another carrier.
    if _is_spatial_envelope(source) and _is_spatial_envelope(target):
        return (
            ("P", augmented, "source envelope retained on an analogous carrier")
            if same_carrier
            else ("R", augmented, "source envelope transferred to a different carrier")
        )

    # On constrained bodies, the a-priori difficult set identifies missing
    # indispensable structure (e.g., bilateral hands).  A spatial trace can
    # explicitly redraw that structure; other motions remain loss cases.
    if physically_difficult and strategy != "trace":
        return "L/D", augmented, "indispensable carrier structure is unavailable"

    # Once a body-specific program has passed the cue's critical O/L/M checks,
    # an event on the same functional carrier is a preserved realization even
    # if joint-axis conventions differ across renderers.
    if same_carrier:
        return "P", augmented, "event retained on an analogous functional carrier"

    if stroke_match(dyn_keys(source), dyn_keys(target)):
        return "R", augmented, "signed stroke transferred to a different carrier"

    if strategy == "trace":
        if _is_spatial_envelope(source) and same_carrier:
            return "P", augmented, "source path retained on an analogous carrier"
        return "R", augmented, "source evidence redrawn as a spatial trace"

    if strategy == "carrier":
        if same_carrier:
            return "P", augmented, "signed stroke retained on an analogous carrier"
        return "R", augmented, "signed stroke transferred to a different carrier"

    if strategy == "additive":
        if _oscillates(source) and _oscillates(target):
            return "R", False, "oscillatory evidence transferred to a different carrier"
        return "L/D", True, "added motion does not retain the required source stroke"

    return "L/D", augmented, "no compatible realization of the required source stroke"


def critical_agreement(platform: str, gt_code: str, llm_code: str, critical: dict) -> tuple[bool, list[str]]:
    checks = {
        "O": score_orientation(platform, gt_code, llm_code),
        "L": score_location(platform, gt_code, llm_code),
        "M": score_movement(platform, gt_code, llm_code),
    }
    misses: list[str] = []
    for key in ("O", "L", "M"):
        if not critical[key]:
            continue
        result = checks[key]
        if result["applicable"] and not result["ok"]:
            # Omitted numeric location fields inherit renderer defaults or a
            # feasible-bank pose.  Treat only an explicit contradictory value
            # as program-level distortion; rendered recognition remains the
            # final validation.
            if key == "L" and "→∅" in result["reason"] and "→" not in result["reason"].replace("→∅", ""):
                continue
            misses.append(f"{key}: {result['reason']}")
        elif not result["applicable"]:
            # An unavailable required channel is acceptable only when the
            # target contains dynamics that may rematerialize it.  The final
            # base_outcome still decides whether that rematerialization exists.
            if not _has_event(llm_code):
                misses.append(f"{key}: required channel unavailable and not rematerialized")
    return not misses, misses


def display(primary: str, augmented: bool) -> str:
    return primary + ("+A" if augmented else "")


def main() -> None:
    gt = load_by_cue(GT_PATH)
    llm = load_by_cue(LLM_PATH)
    critical_map = load_critical()
    difficulty_raw = json.loads(DIFFICULTY_PATH.read_text(encoding="utf-8"))
    difficult = {
        platform: set(difficulty_raw[body]["difficult"])
        for platform, body in DIFFICULTY_KEY.items()
    }
    rows = []
    details = []

    for cue, gt_master in gt.items():
        critical = critical_map[cue]
        source = _code(gt_master["platforms"]["humanoid"])
        signature = "".join(key for key in ("O", "L", "M") if critical[key]) or "M"
        row = {"idx": int(gt_master["idx"]), "cue": cue, "signature": signature}

        for platform in PLATFORMS:
            human_code = _code(gt_master["platforms"][platform])
            llm_code = _code(llm[cue]["platforms"][platform])

            is_difficult = cue in difficult.get(platform, set())
            hp, ha, hr = base_outcome(source, human_code, platform, critical, is_difficult)
            lp, la, lr = base_outcome(source, llm_code, platform, critical, is_difficult)
            agrees, misses = critical_agreement(platform, human_code, llm_code, critical)
            if not agrees:
                lp = "L/D"

            hlabel, llabel = display(hp, ha), display(lp, la)
            row[f"{platform}_human"] = hlabel
            row[f"{platform}_llm"] = llabel
            details.append(
                {
                    "idx": row["idx"],
                    "cue": cue,
                    "platform": platform,
                    "signature": signature,
                    "human": {"label": hlabel, "reason": hr, "code": human_code},
                    "llm": {
                        "label": llabel,
                        "reason": lr,
                        "critical_agreement": agrees,
                        "critical_misses": misses,
                        "code": llm_code,
                    },
                }
            )
        rows.append(row)

    summary = {}
    for platform in PLATFORMS:
        summary[platform] = {}
        for author in ("human", "llm"):
            labels = [row[f"{platform}_{author}"] for row in rows]
            primary = Counter(label.replace("+A", "") for label in labels)
            summary[platform][author] = {
                "P": primary["P"],
                "R": primary["R"],
                "L/D": primary["L/D"],
                "A": sum(label.endswith("+A") for label in labels),
            }

    payload = {
        "status": "provisional program-derived coding",
        "legend": {
            "P": "Preserved",
            "R": "Rematerialized",
            "A": "Augmented (orthogonal flag)",
            "L/D": "Lost or distorted",
        },
        "summary": summary,
        "rows": rows,
        "details": details,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = ["idx", "cue", "signature"] + [
        f"{platform}_{author}"
        for platform in PLATFORMS
        for author in ("human", "llm")
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Transfer outcomes for the 30-cue screening set",
        "",
        "**Status:** provisional program-derived coding. `H / L` denotes human-authored / LLM-authored. ",
        "`P` = Preserved, `R` = Rematerialized, `A` = Augmented, and `L/D` = Lost or Distorted. ",
        "Augmentation is an orthogonal flag, so combinations such as `P+A` and `L/D+A` are allowed.",
        "",
        "The common semantic source is the human-authored G1 program. LLM outputs are additionally checked against the body-specific human program on the cue's required O/L/M channels. The JSON companion records every rule and mismatch. Independent coding and recognition data are still required before these labels are reported as empirical findings.",
        "",
        "## Distribution",
        "",
        "Counts are out of 30 cues. `A` overlaps the three primary outcomes.",
        "",
        "| Robot | Author | P | R | L/D | A |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for platform in PLATFORMS:
        for author, author_label in (("human", "Human"), ("llm", "LLM")):
            counts = summary[platform][author]
            lines.append(
                f"| {PLATFORM_LABEL[platform]} | {author_label} | {counts['P']} | {counts['R']} | {counts['L/D']} | {counts['A']} |"
            )
    lines += [
        "",
        "## Per-cue coding",
        "",
        "| # | Cue | Critical | G1 H / L | TIAGo H / L | IIWA H / L | Stretch H / L | Roomba H / L |",
        "|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        cells = []
        for platform in PLATFORMS:
            cells.append(f"{row[f'{platform}_human']} / {row[f'{platform}_llm']}")
        lines.append(
            f"| {row['idx']} | `{row['cue']}` | {row['signature']} | "
            + " | ".join(cells)
            + " |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{@{}r l c cc cc cc cc cc@{}}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{G1} & \multicolumn{2}{c}{TIAGo} & \multicolumn{2}{c}{IIWA} & \multicolumn{2}{c}{Stretch} & \multicolumn{2}{c}{Roomba} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(l){12-13}",
        r"\# & Cue & Crit. & H & LLM & H & LLM & H & LLM & H & LLM & H & LLM \\",
        r"\midrule",
    ]
    for row in rows:
        cue_tex = str(row["cue"]).replace("_", r"\_")
        cells = []
        for platform in PLATFORMS:
            cells.extend((row[f"{platform}_human"], row[f"{platform}_llm"]))
        tex.append(
            f"{row['idx']} & \\texttt{{{cue_tex}}} & {row['signature']} & "
            + " & ".join(cells)
            + r" \\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Program-derived transfer coding for the 30-cue screening set. H denotes human-authored. P: preserved; R: rematerialized; A: augmented; L/D: lost or distorted. Augmentation is an orthogonal flag.}",
        r"\label{tab:transfer-outcomes-30}",
        r"\end{table*}",
    ]
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
