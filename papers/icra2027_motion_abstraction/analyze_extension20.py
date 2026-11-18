#!/usr/bin/env python3
"""Build the reproducible 50-cue inventory and structural audit for extension20."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
EXT = REPO / "data/seed/agent_fill/embodiment_extension20/cues.json"
CUES90 = REPO / "data/seed/agent_fill/embodiment_short/cues_humanlike90.json"
TAXONOMY = PAPER / "body_part_taxonomy.json"
FEASIBILITY = PAPER / "embodiment_physical_difficulty_90.json"
HARD_NEG = PAPER / "humanlike90_4afc_eval/hard_negatives.json"
PROGRAMS = REPO / "data/results/motion_configs/embodiment_extension20_gemini/embodiment_motion_code_all.json"
RENDER = REPO / "data/results/render/embodiment_extension20_gemini"
OUT = REPO / "data/results/evaluation/embodiment_extension20_structural_audit.json"
TRIALS = REPO / "data/results/evaluation/embodiment_extension20_hardneg_trials.json"
TABLE = PAPER / "tables/cue_inventory_50.tex"
FIG = PAPER / "figures/figure_extension20_structural_audit.png"
FIG_PDF = PAPER / "figures/figure_extension20_structural_audit.pdf"
FAIL_FIG = PAPER / "figures/figure_extension20_failure_examples.png"

PLATFORMS = ("single_arm", "stretch", "roomba")
DISPLAY = {"single_arm": "IIWA", "stretch": "Stretch", "roomba": "Roomba"}
FEAS_KEY = {"single_arm": "Single Arm", "stretch": "Stretch", "roomba": "Roomba"}
MORPH_ORDER = ("Humanoid", "Tiago", "Single Arm", "Stretch", "Roomba")
MORPH_SHORT = {
    "Humanoid": "G1",
    "Tiago": "TIAGo",
    "Single Arm": "IIWA",
    "Stretch": "Stretch",
    "Roomba": "Roomba",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tex(text: str) -> str:
    return text.replace("_", r"\_")


def has_secondary_carrier(code: str, group: str) -> bool:
    nondefault_head = bool(re.search(r"head=(?!center\b)[a-z]+", code))
    nondefault_torso = bool(re.search(r"torso=(?:high|low)", code))
    active_head_torso = bool(re.search(r"move\((?:head|torso),", code))
    base_motion = "path(base," in code
    if group == "One arm / hand":
        return nondefault_head or nondefault_torso or active_head_torso or base_motion
    if group == "Head / torso + one arm":
        return base_motion
    if group == "Two arms / hands":
        return nondefault_head or nondefault_torso or active_head_torso or base_motion
    return False


def transfer_mode(platform: str, group: str, feasible: bool, code: str) -> str:
    if not feasible:
        return "At-risk substitute"
    if platform == "roomba":
        return "Rematerialized"
    if has_secondary_carrier(code, group):
        return "Augmented"
    if platform == "single_arm":
        return "Preserved" if group == "One arm / hand" else "Carrier-transferred"
    if group in {"Head / torso", "Head / torso + one arm", "One arm / hand"}:
        return "Preserved"
    return "Carrier-transferred"


def build_table(rows: list[dict]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["required_part"], []).append(row)
    order = ("Head / torso", "Head / torso + one arm", "One arm / hand", "Two arms / hands")
    blocks = []
    for number, group in enumerate(order, start=1):
        group_rows = grouped[group]
        body = []
        for row in group_rows:
            morphs = ", ".join(MORPH_SHORT[m] for m in row["feasible_morphologies"])
            body.append(f'{row["idx"]} & \\texttt{{{tex(row["cue"])}}} & {morphs} \\\\')
        blocks.append(
            "\n".join(
                [
                    r"\begin{table}[t]",
                    r"\centering",
                    r"\scriptsize",
                    r"\setlength{\tabcolsep}{3pt}",
                    r"\renewcommand{\arraystretch}{1.05}",
                    r"\begin{tabularx}{\columnwidth}{@{}r>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.3cm}@{}}",
                    r"\toprule",
                    r"\textbf{ID} & \textbf{Cue} & \textbf{A-priori feasible morphologies} \\",
                    r"\midrule",
                    *body,
                    r"\bottomrule",
                    r"\end{tabularx}",
                    f"\\caption{{{group} cues in the 50-cue evaluation inventory ($n={len(group_rows)}$). Feasibility follows the carrier-preservation rule in Sec.~\\ref{{sec:experiment}} and does not imply recognition.}}",
                    f"\\label{{tab:cue-inventory-50-{number}}}",
                    r"\end{table}",
                ]
            )
        )
    TABLE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def gif_path(platform: str, idx: int, cue: str) -> Path:
    stem = f"emb_g{idx:02d}_{cue}.gif"
    if platform == "single_arm":
        return RENDER / "manipulator/gif/IIWA" / stem
    return RENDER / platform / "gif" / stem


def frame_strip(path: Path, width: int = 760, height: int = 180) -> Image.Image:
    gif = Image.open(path)
    n = getattr(gif, "n_frames", 1)
    picks = sorted(set((0, max(0, n // 2), max(0, n - 1))))
    frames = []
    for frame_idx in picks:
        gif.seek(frame_idx)
        frame = gif.convert("RGB")
        frame.thumbnail((width // 3 - 10, height - 10), Image.Resampling.LANCZOS)
        frames.append(frame)
    canvas = Image.new("RGB", (width, height), "white")
    slot = width // 3
    for i, frame in enumerate(frames):
        x = i * slot + (slot - frame.width) // 2
        y = (height - frame.height) // 2
        canvas.paste(frame, (x, y))
    return canvas


def build_failure_figure() -> None:
    cases = [
        ("Preserved", "stretch", 40, "fistbump_invite", "Arm extension retains the original carrier and stroke."),
        ("Rematerialized", "roomba", 65, "continue_on_forward_roll", "A circular hand roll becomes a circular base path."),
        ("Augmented", "stretch", 43, "visor_search", "The raised arm adds a visor while the head performs the scan."),
        ("At-risk substitute", "roomba", 90, "clap_applause", "Forward-backward oscillation cannot retain two-hand contact or symmetry."),
    ]
    width, row_h = 1200, 245
    canvas = Image.new("RGB", (width, row_h * len(cases) + 30), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    colors = {
        "Preserved": "#287D4F",
        "Rematerialized": "#6C4AB6",
        "Augmented": "#2673B8",
        "At-risk substitute": "#B43A3A",
    }
    for i, (mode, platform, idx, cue, explanation) in enumerate(cases):
        y = 15 + i * row_h
        strip = frame_strip(gif_path(platform, idx, cue), width=730, height=205)
        canvas.paste(strip, (450, y + 20))
        draw.rounded_rectangle((20, y + 16, 425, y + 225), radius=16, fill="#F5F7FA", outline=colors[mode], width=4)
        draw.text((42, y + 38), mode, fill=colors[mode], font=font)
        draw.text((42, y + 78), f"{DISPLAY[platform]} · {cue}", fill="#17202A", font=small)
        words = explanation.split()
        lines, line = [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=small) > 350 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        for j, text in enumerate(lines):
            draw.text((42, y + 126 + j * 25), text, fill="#4C5663", font=small)
    FAIL_FIG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(FAIL_FIG)


def main() -> None:
    ext = load(EXT)
    cues90 = load(CUES90)
    cue_by_idx = {int(row["idx"]): row for row in cues90}
    selected_ids = list(range(1, 31)) + [int(row["idx"]) for row in ext]
    taxonomy = load(TAXONOMY)["cue_to_cat"]
    feasibility = load(FEASIBILITY)
    feasible_sets = {name: set(feasibility["by_robot"][name]["feasible"]) for name in MORPH_ORDER}
    matrix = {row["cue"]: row for row in feasibility["matrix"]}
    programs = load(PROGRAMS)

    inventory = []
    for idx in selected_ids:
        cue = cue_by_idx[idx]["cue"]
        inventory.append(
            {
                "idx": idx,
                "cue": cue,
                "required_part": taxonomy[cue],
                "feasible_morphologies": [m for m in MORPH_ORDER if matrix[cue][m] == "feasible"],
            }
        )
    build_table(inventory)

    base_cues = {cue_by_idx[idx]["cue"] for idx in range(1, 31)}
    ext_cues = {row["cue"] for row in ext}
    expansion = {}
    for key, display in FEAS_KEY.items():
        base_n = len(base_cues & feasible_sets[display])
        added_n = len(ext_cues & feasible_sets[display])
        expansion[key] = {
            "base_feasible_n": base_n,
            "added_feasible_n": added_n,
            "total_feasible_n": base_n + added_n,
            "relative_expansion_pct": round(100 * added_n / base_n, 1) if base_n else None,
            "coverage_of_50_pct": round(100 * (base_n + added_n) / 50, 1),
        }

    rows = []
    primitive_presence = {platform: Counter() for platform in PLATFORMS}
    mode_counts = {platform: Counter() for platform in PLATFORMS}
    for cue_row in programs:
        cue = cue_row["cue"]
        group = taxonomy[cue]
        for platform in PLATFORMS:
            item = cue_row["platforms"][platform]
            types = [movement["type"] for movement in item.get("movements", [])]
            for primitive in set(types):
                primitive_presence[platform][primitive] += 1
            feasible = cue in feasible_sets[FEAS_KEY[platform]]
            mode = transfer_mode(platform, group, feasible, item["source_code"])
            mode_counts[platform][mode] += 1
            rows.append(
                {
                    "idx": cue_row["idx"],
                    "cue": cue,
                    "platform": platform,
                    "required_part": group,
                    "a_priori_feasible": feasible,
                    "transfer_mode": mode,
                    "primitive_sequence": types,
                    "source_code": item["source_code"],
                }
            )

    hardneg_by_idx = {int(row["idx"]): row for row in load(HARD_NEG)["rows"]}
    rng = random.Random(20260915)
    trials = []
    for cue_row in ext:
        idx, cue = int(cue_row["idx"]), cue_row["cue"]
        negatives = list(hardneg_by_idx[idx]["hybrid"])
        for platform in PLATFORMS:
            choices = [cue, *negatives]
            rng.shuffle(choices)
            trials.append(
                {
                    "trial_id": f"{platform}.c{idx:02d}",
                    "platform": platform,
                    "idx": idx,
                    "target": cue,
                    "choices": choices,
                    "answer_index": choices.index(cue),
                    "negatives": negatives,
                    "foil_strategy": "nearest Motion-Program foil plus two meaning-neighbor foils",
                    "seed": 20260915,
                    "gif": str(gif_path(platform, idx, cue)),
                }
            )
    rng.shuffle(trials)
    for order, trial in enumerate(trials, start=1):
        trial["presentation_order"] = order
    TRIALS.parent.mkdir(parents=True, exist_ok=True)
    TRIALS.write_text(json.dumps(trials, indent=2) + "\n", encoding="utf-8")

    payload = {
        "scope": "20 extension cues x 3 constrained catalog morphologies",
        "interpretation_limit": "Structural audit only; transfer-mode labels and feasibility do not establish human recognition.",
        "inventory_50": inventory,
        "feasible_expansion": expansion,
        "primitive_presence": {p: dict(c) for p, c in primitive_presence.items()},
        "transfer_mode_counts": {p: dict(c) for p, c in mode_counts.items()},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    colors = {"single_arm": "#376FA6", "stretch": "#C56A35", "roomba": "#4E8B57"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    names = list(PLATFORMS)
    x = np.arange(len(names))
    base = [expansion[p]["base_feasible_n"] for p in names]
    added = [expansion[p]["added_feasible_n"] for p in names]
    axes[0].bar(x, base, color="#BCC6D1", label="Feasible in original 30")
    axes[0].bar(x, added, bottom=base, color=[colors[p] for p in names], label="Feasible among added 20")
    axes[0].set_xticks(x, [DISPLAY[p] for p in names])
    axes[0].set_ylabel("A-priori feasible cues")
    axes[0].set_ylim(0, 32)
    axes[0].set_title("Feasible inventory expands unevenly")
    for i, p in enumerate(names):
        axes[0].text(i, base[i] + added[i] + 0.7, f'+{expansion[p]["relative_expansion_pct"]:.0f}%', ha="center", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    modes = ["Preserved", "Carrier-transferred", "Rematerialized", "Augmented", "At-risk substitute"]
    mode_colors = ["#287D4F", "#A37A27", "#6C4AB6", "#2673B8", "#B43A3A"]
    left = np.zeros(len(names))
    for mode, color in zip(modes, mode_colors):
        values = np.array([mode_counts[p][mode] for p in names])
        axes[1].barh([DISPLAY[p] for p in names], values, left=left, color=color, label=mode)
        for i, value in enumerate(values):
            if value >= 2:
                axes[1].text(left[i] + value / 2, i, str(int(value)), ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += values
    axes[1].set_xlim(0, 20)
    axes[1].set_xlabel("Generated extension cues")
    axes[1].set_title("Structural transfer modes")
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=220)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    build_failure_figure()

    print(json.dumps({"audit": str(OUT), "trials": str(TRIALS), "table": str(TABLE), "figure": str(FIG), "failure_figure": str(FAIL_FIG)}, indent=2))


if __name__ == "__main__":
    main()
