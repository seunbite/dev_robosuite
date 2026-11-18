#!/usr/bin/env python3
"""Program-derived transfer audit for 50 cues x 3 non-anthropomorphic robots.

The audit mirrors Sec. 3.3's two-axis taxonomy:
  strategy: Preserved / Carrier-transferred / Rematerialized
  consequence: Augmented / Maintained / Lost

These labels describe program structure under an a-priori morphology-feasibility
prior.  They are not human-recognition measurements.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "motion_code"))

from build_transfer_outcome_30 import carrier_families, load_critical  # noqa: E402
from classify_rematerialization import classify, has_dynamics, is_envelope  # noqa: E402
from score_nonhumanoid_olm import _code, load_by_cue  # noqa: E402

GT30 = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
LLM30 = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
EXT20 = REPO / "data/results/motion_configs/embodiment_extension20_gemini/embodiment_motion_code_all.json"
HUMANLIKE = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike/embodiment_motion_code_all.json"
HUMANLIKE_FALLBACK = REPO / "data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90/embodiment_motion_code_all.json"
TAXONOMY = HERE / "body_part_taxonomy.json"
FEASIBILITY = HERE / "embodiment_physical_difficulty_90.json"

OUT_JSON = HERE / "transfer_outcomes_50.json"
OUT_CSV = HERE / "transfer_outcomes_50.csv"
OUT_MD = HERE / "transfer_outcomes_50.md"
OUT_TEX = HERE / "tables/nonanthropomorphic_transfer_50_summary.tex"
OUT_FIG = HERE / "figures/figure_nonanthropomorphic_transfer_50_stacked.pdf"
OUT_FIG_PNG = HERE / "figures/figure_nonanthropomorphic_transfer_50_stacked.png"
OUT_FIG_SVG = HERE / "figures/figure_nonanthropomorphic_transfer_50_stacked.svg"
# Strategy panel only: the consequence panel restates the a-priori feasibility prior.
OUT_FIG_STRATEGY = HERE / "figures/figure_nonanthropomorphic_strategy_50.pdf"
# Human-authored programs exist only for the 30-cue screen; the LLM-30 figure is its matched comparison.
OUT_FIG_HUMAN30 = HERE / "figures/figure_nonanthropomorphic_transfer_human30_stacked.pdf"
OUT_FIG_LLM30 = HERE / "figures/figure_nonanthropomorphic_transfer_llm30_stacked.pdf"
OUT_HUMAN_LLM30_JSON = HERE / "transfer_outcomes_human_llm_30.json"
OUT_HUMAN_LLM30_TEX = HERE / "tables/nonanthropomorphic_human_llm_30_summary.tex"
RECOGNITION_JSON = REPO / "data/results/evaluation/embodiment_cue_id/cueid_human_cue_identification_with_traj_baselines.json"

PLATFORMS = ("single_arm", "stretch", "roomba")
DISPLAY = {"single_arm": "IIWA", "stretch": "Stretch", "roomba": "Roomba"}
FEAS_KEY = {"single_arm": "Single Arm", "stretch": "Stretch", "roomba": "Roomba"}
STRATEGIES = ("Preserved", "Carrier-transferred", "Rematerialized")
CONSEQUENCES = ("Augmented", "Maintained", "Lost")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def program_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    old = load_by_cue(LLM30)
    ext = load_by_cue(EXT20)
    rows = sorted([*old.values(), *ext.values()], key=lambda row: int(row["idx"]))
    if len(rows) != 50:
        raise ValueError(f"expected 50 cues, found {len(rows)}")
    return rows, {**old, **ext}


def source_programs() -> tuple[dict[str, str], dict[str, str]]:
    """Use human G1 GT where available; otherwise use the LLM G1 realization."""
    refs: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for cue, row in load_by_cue(GT30).items():
        refs[cue] = _code(row["platforms"]["humanoid"])
        provenance[cue] = "human-authored G1 reference"
    for cue, row in load_by_cue(HUMANLIKE).items():
        if cue in refs or "humanoid" not in row.get("platforms", {}):
            continue
        refs[cue] = _code(row["platforms"]["humanoid"])
        provenance[cue] = "LLM-authored G1 reference (no human extension reference)"
    for cue, row in load_by_cue(HUMANLIKE_FALLBACK).items():
        if cue in refs or "humanoid" not in row.get("platforms", {}):
            continue
        refs[cue] = _code(row["platforms"]["humanoid"])
        provenance[cue] = "Qwen-authored G1 fallback reference (no Gemini or human extension reference)"
    return refs, provenance


def has_augmentation(code: str, required_part: str, platform: str) -> bool:
    """Conservative, morphology-aware structural augmentation flag.

    Augmentation requires motion on at least two independently controllable
    carrier families. Multiple strokes on one arm or on the base do not count.
    For head/torso + one-arm cues, those two channels satisfy the cue itself;
    a third dynamic family is required before calling the program augmented.
    """
    carriers = carrier_families(code)
    if len(carriers) < 2:
        return False
    if required_part == "Head / torso + one arm":
        secondary = carriers & {"head", "torso", "base"}
        return "upper_limb" in carriers and len(secondary) >= 2
    return True


def score_strategy(ref: str, target: str) -> tuple[str, str]:
    raw = classify(ref, target)
    source_carriers = carrier_families(ref)
    target_carriers = carrier_families(target)
    same_carrier = bool(source_carriers & target_carriers)
    if raw == "trace":
        if is_envelope(ref) and is_envelope(target):
            if same_carrier:
                return "Preserved", "the same spatial envelope remains on an analogous carrier"
            return "Carrier-transferred", "the same spatial envelope moves to a different carrier"
        return "Rematerialized", "source evidence is redrawn as a spatial trace"
    if raw == "carrier":
        if same_carrier:
            return "Preserved", "a matching signed stroke remains on an analogous carrier"
        return "Carrier-transferred", "a matching signed stroke moves to a different carrier"
    if raw == "additive":
        return "Rematerialized", "the target uses a different dynamic structure"
    if not has_dynamics(ref) and not has_dynamics(target) and same_carrier:
        return "Preserved", "static evidence remains on an analogous carrier"
    return "Rematerialized", "the source event is not directly preserved"


def score_consequence(
    *, cue: str, platform: str, required_part: str, feasible: bool,
    ref: str, target: str, critical: dict[str, Any],
) -> tuple[str, str]:
    if not feasible:
        return "Lost", "an indispensable carrier or relation is unavailable under the a-priori feasibility rule"
    if bool(critical.get("M")) and has_dynamics(ref) and not has_dynamics(target):
        return "Lost", "the reference requires a dynamic event but the target program contains none"
    if has_augmentation(target, required_part, platform):
        return "Augmented", "the program recruits an additional independently controllable carrier"
    return "Maintained", "the required structure is feasible and no indispensable event is absent"


def pct(n: int, total: int = 50) -> str:
    return f"{100.0 * n / total:.0f}\\%"


# Validated with the dataviz palette checker (adjacent CVD dE >= 9); every
# segment >= 3% carries a direct label, which relieves the low-contrast fills.
STRATEGY_COLORS = {
    "Preserved": "#2A78D6",
    "Carrier-transferred": "#EDA100",
    "Rematerialized": "#4A3AA7",
}
CONSEQUENCE_COLORS = {
    "Augmented": "#0F7F5C",
    "Maintained": "#52B788",
    "Lost": "#E34948",
}
DARK_FILLS = {"#2A78D6", "#4A3AA7", "#0F7F5C", "#E34948"}
COLUMN_WIDTH_IN = 245.71811 / 72.27  # ieeeconf \columnwidth


def build_stacked_figure(
    strategy_counts: dict[str, Counter],
    consequence_counts: dict[str, Counter] | None = None,
    out_pdf: Path = OUT_FIG,
    n_programs: int = 50,
    author: str = "LLM-authored",
) -> None:
    """Draw 100%-stacked horizontal bars at IEEE column width (one or two panels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1F2328", "#5B6470"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.edgecolor": muted,
        "xtick.color": muted,
        "ytick.color": ink,
    })
    panels = [
        ("Realization strategy", STRATEGIES, strategy_counts, STRATEGY_COLORS),
        ("Evidence consequence", CONSEQUENCES, consequence_counts, CONSEQUENCE_COLORS),
    ][: 1 if consequence_counts is None else 2]
    if len(panels) == 2:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.0), sharex=True)
        fig.subplots_adjust(left=0.145, right=0.972, top=0.90, bottom=0.17, hspace=0.42)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.08))
        fig.subplots_adjust(left=0.145, right=0.972, top=0.81, bottom=0.265)
        axes = [ax]
    rows = list(reversed(PLATFORMS))  # IIWA on top
    for ax, (title, labels, counts, colors) in zip(axes, panels):
        for y, platform in enumerate(rows):
            left = 0
            for label in labels:
                count = counts[platform][label]
                if not count:
                    continue
                value = 100.0 * count / n_programs
                shown = round(value)
                ax.barh(y, value, left=left, height=0.74, color=colors[label],
                        edgecolor="white", linewidth=1.0)
                if value >= 3:
                    ax.text(left + value / 2, y, f"{shown}%" if value >= 8 else str(shown),
                            ha="center", va="center", fontsize=7 if value >= 8 else 6,
                            fontweight="bold",
                            color="white" if colors[label].upper() in DARK_FILLS else ink)
                left += value
        ax.set_yticks(range(len(rows)), [DISPLAY[p] for p in rows], fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=2, pad=1.5, labelsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        # Title flush with the figure's left edge; legend shares its row on the right.
        fig.text(0.004, ax.get_position().y1 + 0.012, title, ha="left", va="bottom",
                 fontsize=7.5, fontweight="bold", color=ink)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=3,
                  frameon=False, fontsize=6.5, handlelength=0.8, handleheight=0.8,
                  handletextpad=0.3, columnspacing=0.6, borderaxespad=0, borderpad=0.1)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="x", length=0)
    axes[-1].set_xticks([0, 25, 50, 75, 100])
    axes[-1].set_xlabel(f"Share of {n_programs} {author} Motion Programs (%)", fontsize=7, color=muted, labelpad=1.5)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".svg"))
    fig.savefig(out_pdf.with_suffix(".png"), dpi=400)
    plt.close(fig)


def audit_counts(target_path: Path, cues: list[str]) -> tuple[dict[str, Counter], dict[str, Counter], dict[str, Counter]]:
    """Strategy/consequence counts for one program source, using the same rules as main()."""
    targets = load_by_cue(target_path)
    refs, _ = source_programs()
    taxonomy = load(TAXONOMY)["cue_to_cat"]
    feasibility = load(FEASIBILITY)
    critical = load_critical()
    strategy_counts = {p: Counter() for p in PLATFORMS}
    consequence_counts = {p: Counter() for p in PLATFORMS}
    augmentation_counts = {p: Counter() for p in PLATFORMS}
    for cue in cues:
        for platform in PLATFORMS:
            target = _code(targets[cue]["platforms"][platform])
            strategy_counts[platform][score_strategy(refs[cue], target)[0]] += 1
            augmentation_counts[platform]["Augmented"] += int(
                has_augmentation(target, taxonomy[cue], platform)
            )
            consequence_counts[platform][score_consequence(
                cue=cue, platform=platform, required_part=taxonomy[cue],
                feasible=cue in feasibility["by_robot"][FEAS_KEY[platform]]["feasible"],
                ref=refs[cue], target=target, critical=critical[cue],
            )[0]] += 1
    return strategy_counts, consequence_counts, augmentation_counts


def main() -> None:
    if "--human30" in sys.argv:
        cues = sorted(load_by_cue(GT30), key=lambda c: int(load_by_cue(GT30)[c]["idx"]))
        human = audit_counts(GT30, cues)
        llm = audit_counts(LLM30, cues)
        build_stacked_figure(*human[:2], out_pdf=OUT_FIG_HUMAN30, n_programs=len(cues), author="human-authored")
        build_stacked_figure(*llm[:2], out_pdf=OUT_FIG_LLM30, n_programs=len(cues), author="LLM-authored")
        recognition = load(RECOGNITION_JSON)
        platform_index = {name: i for i, name in enumerate(recognition["platforms"])}
        payload = {
            "status": "matched 30-cue comparison; recognition and structural coding are separate measures",
            "recognition_source": str(RECOGNITION_JSON.relative_to(REPO)),
            "human30": {p: {"strategy": dict(human[0][p]), "consequence": dict(human[1][p]), "augmentation_candidates": human[2][p]["Augmented"]} for p in PLATFORMS},
            "llm30": {p: {"strategy": dict(llm[0][p]), "consequence": dict(llm[1][p]), "augmentation_candidates": llm[2][p]["Augmented"]} for p in PLATFORMS},
            "figures": [str(OUT_FIG_HUMAN30), str(OUT_FIG_LLM30)],
        }
        for label, path in (("human30", GT30), ("llm30", LLM30)):
            targets = load_by_cue(path)
            for p in PLATFORMS:
                payload[label][p]["augmentation_cues"] = [
                    cue for cue in cues
                    if has_augmentation(
                        _code(targets[cue]["platforms"][p]),
                        load(TAXONOMY)["cue_to_cat"][cue], p,
                    )
                ]
        payload["recognition_percent"] = {
            p: {
                "human_authored": recognition["human_authored"][platform_index[p]],
                "llm_authored": recognition["ours_gemini_2_5_pro"][platform_index[p]],
            }
            for p in PLATFORMS
        }
        OUT_HUMAN_LLM30_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        tex = [
            r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{4.5pt}", r"\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
            r"\textbf{Robot} & \textbf{Author} & \textbf{Recognition} & \textbf{Remat.} & \textbf{Aug.} \\",
            r"\midrule",
        ]
        for p in PLATFORMS:
            for row_i, (author, counts) in enumerate((("Human", human), ("LLM", llm))):
                prefix = f"\\multirow{{2}}{{*}}{{{DISPLAY[p]}}}" if row_i == 0 else ""
                rec = payload["recognition_percent"][p]["human_authored" if author == "Human" else "llm_authored"]
                remat = 100.0 * counts[0][p]["Rematerialized"] / len(cues)
                aug = 100.0 * counts[2][p]["Augmented"] / len(cues)
                tex.append(f"{prefix} & {author} & {rec:.1f} & {remat:.1f} & {aug:.1f} \\\\")
            if p != PLATFORMS[-1]:
                tex.append(r"\addlinespace[1pt]")
        tex += [
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Matched 30-cue comparison on non-humanoid robots (\%). Recognition is the reviewed human yes/no judgment. Rematerialization is a mutually exclusive realization strategy. Augmentation is an independent structural flag requiring motion on an additional controllable carrier; neither structural label is a recognition outcome.}",
            r"\label{tab:nonanthro-human-llm-30}", r"\end{table}",
        ]
        OUT_HUMAN_LLM30_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
        payload["table"] = str(OUT_HUMAN_LLM30_TEX)
        print(json.dumps(payload, indent=2))
        return

    if "--figure-only" in sys.argv:
        summary = load(OUT_JSON)["summary"]
        build_stacked_figure(
            {p: Counter(summary[p]["strategy"]) for p in PLATFORMS},
            {p: Counter(summary[p]["consequence"]) for p in PLATFORMS},
        )
        build_stacked_figure({p: Counter(summary[p]["strategy"]) for p in PLATFORMS}, out_pdf=OUT_FIG_STRATEGY)
        print(json.dumps({"figure": str(OUT_FIG), "strategy_figure": str(OUT_FIG_STRATEGY)}, indent=2))
        return

    rows, by_cue = program_rows()
    refs, provenance = source_programs()
    taxonomy = load(TAXONOMY)["cue_to_cat"]
    feasibility = load(FEASIBILITY)
    feasible_sets = {
        platform: set(feasibility["by_robot"][FEAS_KEY[platform]]["feasible"])
        for platform in PLATFORMS
    }
    critical = load_critical()

    details: list[dict[str, Any]] = []
    strategy_counts = {p: Counter() for p in PLATFORMS}
    consequence_counts = {p: Counter() for p in PLATFORMS}
    augmentation_counts = Counter()
    by_body_part = {
        p: {cat: Counter() for cat in sorted(set(taxonomy.values()))}
        for p in PLATFORMS
    }

    for row in rows:
        cue = str(row["cue"])
        ref = refs.get(cue)
        if not ref:
            raise KeyError(f"missing humanoid reference for {cue}")
        required_part = taxonomy[cue]
        for platform in PLATFORMS:
            target = _code(by_cue[cue]["platforms"][platform])
            feasible = cue in feasible_sets[platform]
            strategy, strategy_reason = score_strategy(ref, target)
            consequence, consequence_reason = score_consequence(
                cue=cue,
                platform=platform,
                required_part=required_part,
                feasible=feasible,
                ref=ref,
                target=target,
                critical=critical[cue],
            )
            augmentation_candidate = has_augmentation(target, required_part, platform)
            strategy_counts[platform][strategy] += 1
            consequence_counts[platform][consequence] += 1
            augmentation_counts[platform] += int(augmentation_candidate)
            by_body_part[platform][required_part][consequence] += 1
            details.append(
                {
                    "idx": int(row["idx"]),
                    "cue": cue,
                    "platform": platform,
                    "required_part": required_part,
                    "reference_provenance": provenance[cue],
                    "a_priori_feasible": feasible,
                    "strategy": strategy,
                    "strategy_reason": strategy_reason,
                    "consequence": consequence,
                    "augmentation_candidate": augmentation_candidate,
                    "consequence_reason": consequence_reason,
                    "reference_code": ref,
                    "target_code": target,
                }
            )

    summary = {
        p: {
            "strategy": dict(strategy_counts[p]),
            "consequence": dict(consequence_counts[p]),
            "augmentation_candidates": augmentation_counts[p],
            "by_required_body_part": {k: dict(v) for k, v in by_body_part[p].items()},
        }
        for p in PLATFORMS
    }
    payload = {
        "status": "automatic program-derived structural audit; not a recognition result",
        "scope": "50 cues x IIWA, Stretch, and Roomba",
        "reference_note": "Cues 1--30 use human-authored G1 programs; extension cues use LLM-authored G1 programs because human references are unavailable.",
        "taxonomy": {
            "strategy": list(STRATEGIES),
            "consequence": list(CONSEQUENCES),
        },
        "summary": summary,
        "details": details,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "idx", "cue", "platform", "required_part", "a_priori_feasible",
            "strategy", "consequence", "strategy_reason", "consequence_reason",
            "reference_provenance", "reference_code", "target_code",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    md = [
        "# 50-cue non-anthropomorphic transfer audit", "",
        "Automatic program-derived coding; these labels do not establish human recognition.", "",
        "## Compact result", "",
        "| Robot | Preserved | Carrier-transferred | Rematerialized | Augmented | Maintained | Lost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in PLATFORMS:
        s, c = strategy_counts[p], consequence_counts[p]
        md.append(
            f"| {DISPLAY[p]} | {s['Preserved']} | {s['Carrier-transferred']} | {s['Rematerialized']} | "
            f"{c['Augmented']} | {c['Maintained']} | {c['Lost']} |"
        )
    md += ["", "## Per-program coding", "",
           "| ID | Cue | Robot | Required body parts | Feasible | Strategy | Consequence | Reason |",
           "|---:|---|---|---|:---:|---|---|---|"]
    for d in details:
        md.append(
            f"| {d['idx']} | `{d['cue']}` | {DISPLAY[d['platform']]} | {d['required_part']} | "
            f"{'Y' if d['a_priori_feasible'] else 'N'} | {d['strategy']} | {d['consequence']} | "
            f"{d['consequence_reason']} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3.5pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}lrrr|rrr@{}}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Realization strategy}} & \multicolumn{3}{c}{\textbf{Evidence consequence}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"\textbf{Robot} & \textbf{Pres.} & \textbf{Trans.} & \textbf{Remat.} & \textbf{Aug.} & \textbf{Maint.} & \textbf{Lost} \\",
        r"\midrule",
    ]
    for p in PLATFORMS:
        s, c = strategy_counts[p], consequence_counts[p]
        tex.append(
            f"{DISPLAY[p]} & {s['Preserved']} & {s['Carrier-transferred']} & {s['Rematerialized']} & "
            f"{c['Augmented']} & {c['Maintained']} & {c['Lost']} \\\\"
        )
    tex += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Automatic program-derived coding of 50 LLM-authored cues on each non-anthropomorphic robot. Strategy and consequence are separate, mutually exclusive axes; each triplet sums to 50. These structural labels do not measure human recognition.}",
        r"\label{tab:nonanthro-transfer-50}", r"\end{table}",
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    build_stacked_figure(strategy_counts, consequence_counts)
    build_stacked_figure(strategy_counts, out_pdf=OUT_FIG_STRATEGY)
    print(json.dumps({"json": str(OUT_JSON), "csv": str(OUT_CSV), "md": str(OUT_MD), "tex": str(OUT_TEX), "figure": str(OUT_FIG), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
