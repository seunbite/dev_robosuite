#!/usr/bin/env python3
"""Compare structural complexity of matched human- and LLM-authored programs.

Program length is the number of top-level primitive calls.  Attribute count is
the number of explicitly authored arguments across those calls, including
positional carrier/shape arguments but excluding parser-added defaults.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
SOURCES = {
    "Human": ROOT
    / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json",
    "LLM": ROOT
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json",
}
ROBOTS = {"single_arm": "IIWA", "stretch": "Stretch", "roomba": "Roomba"}
OUT_JSON = PAPER / "program_complexity_human_llm_30.json"
OUT_TEX = PAPER / "tables/nonanthropomorphic_human_llm_30_summary.tex"
OUT_FAMILY_TEX = PAPER / "tables/nonanthropomorphic_motion_family_30.tex"


def primitive_calls(source: str) -> list[tuple[str, str]]:
    """Return names and argument strings for top-level function calls."""
    calls: list[tuple[str, str]] = []
    i = 0
    while i < len(source):
        match = re.search(r"\b[A-Za-z_]\w*\s*\(", source[i:])
        if not match:
            break
        open_pos = i + match.end() - 1
        depth = 1
        j = open_pos + 1
        while j < len(source) and depth:
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
            j += 1
        if depth:
            raise ValueError(f"Unbalanced program: {source!r}")
        name = source[i + match.start() : open_pos].strip()
        calls.append((name, source[open_pos + 1 : j - 1]))
        i = j
    return calls


def count_top_level_arguments(arguments: str) -> int:
    if not arguments.strip():
        return 0
    depth = 0
    count = 1
    for char in arguments:
        if char in "[({":
            depth += 1
        elif char in "]) }".replace(" ", ""):
            depth -= 1
        elif char == "," and depth == 0:
            count += 1
    return count


def motion_family(name: str, arguments: str) -> str | None:
    """Assign each dynamic primitive to one mutually exclusive motion family."""
    compact = re.sub(r"\s+", "", arguments.lower())
    repetition = re.search(r"(?:rep|repetition)=(\d+)", compact)
    oscillates = bool(repetition and int(repetition.group(1)) > 1)
    oscillates = oscillates or "±" in compact or "+/-" in compact
    for match in re.finditer(r"\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]", compact):
        oscillates = oscillates or float(match.group(1)) * float(match.group(2)) < 0

    if name == "move":
        return "Oscillation" if oscillates else "Joint move"
    if name == "path":
        if oscillates:
            return "Oscillation"
        if re.search(r"(^|,)arc(,|$)", compact):
            return "Arc"
        if re.search(r"(^|,)line(,|$)", compact):
            return "Line"
        return "Other path"
    return None


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


records: list[dict[str, object]] = []
cue_sets: dict[str, set[str]] = {}
for author, path in SOURCES.items():
    rows = json.loads(path.read_text())
    cue_sets[author] = {row["cue"] for row in rows}
    for row in rows:
        for platform, robot in ROBOTS.items():
            source = row["platforms"][platform]["source_code"]
            calls = primitive_calls(source)
            families = [family for name, args in calls if (family := motion_family(name, args))]
            records.append(
                {
                    "cue": row["cue"],
                    "robot": robot,
                    "author": author,
                    "program_length": len(calls),
                    "explicit_attributes": sum(count_top_level_arguments(args) for _, args in calls),
                    "motion_families": families,
                    "source_code": source,
                }
            )

if cue_sets["Human"] != cue_sets["LLM"] or len(cue_sets["Human"]) != 30:
    raise ValueError("Expected the same 30 cues in both authoring conditions")

summary: list[dict[str, object]] = []
family_summary: list[dict[str, object]] = []
for robot in ROBOTS.values():
    for author in SOURCES:
        subset = [r for r in records if r["robot"] == robot and r["author"] == author]
        summary.append(
            {
                "robot": robot,
                "author": author,
                "n": len(subset),
                "program_length": describe([int(r["program_length"]) for r in subset]),
                "explicit_attributes": describe([int(r["explicit_attributes"]) for r in subset]),
            }
        )
        family_counts = {
            family: sum(r["motion_families"].count(family) for r in subset)
            for family in ("Oscillation", "Line", "Arc", "Joint move", "Other path")
        }
        dynamic_total = sum(family_counts.values())
        family_summary.append(
            {
                "robot": robot,
                "author": author,
                "dynamic_primitives": dynamic_total,
                "counts": family_counts,
                "percentages": {
                    key: 100 * value / dynamic_total for key, value in family_counts.items()
                },
                "distinct_families_per_program": statistics.mean(
                    len(set(r["motion_families"])) for r in subset
                ),
            }
        )

OUT_JSON.write_text(json.dumps({"definitions": {
    "program_length": "Number of top-level primitive calls, including pose, move, path, and hold.",
    "explicit_attributes": "Number of explicitly authored positional and named arguments across primitive calls; parser-added defaults are excluded.",
}, "summary": summary, "motion_family_summary": family_summary, "per_cue": records}, indent=2) + "\n")

rows = []
types_per_cue = {
    (item["robot"], item["author"]): item["distinct_families_per_program"]
    for item in family_summary
}
for item in summary:
    length = item["program_length"]
    attrs = item["explicit_attributes"]
    robot = item["robot"] if item["author"] == "Human" else ""
    rows.append(
        f"{robot} & {item['author']} & "
        f"${length['mean']:.1f} \\pm {length['sd']:.1f}$ & "
        f"${attrs['mean']:.1f} \\pm {attrs['sd']:.1f}$ & "
        f"${types_per_cue[(item['robot'], item['author'])]:.2f}$ \\\\"
    )
    if item["author"] == "LLM" and item["robot"] != "Roomba":
        rows.append("\\addlinespace[1pt]")

table = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Robot} & \textbf{Author} & \textbf{Program length} & \textbf{Attributes/program} & \textbf{Types/Cue} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Structural complexity and compositional diversity of matched human- and LLM-authored Motion Programs ($n=30$ cues per row; mean $\pm$ SD). Program length counts top-level primitives. Attributes/program sums all explicitly written positional and named arguments; parser-inserted defaults are excluded. Types/Cue is the mean number of distinct dynamic motion families---oscillation, line, arc, and one-way joint motion---used in one program; static \texttt{pose} and \texttt{hold} primitives are excluded.}
\label{tab:nonanthro-human-llm-30}
\end{table}
"""
OUT_TEX.write_text(table)

family_rows = []
for item in family_summary:
    robot = item["robot"] if item["author"] == "Human" else ""
    pct = item["percentages"]
    family_rows.append(
        f"{robot} & {item['author']} & "
        f"{pct['Oscillation']:.1f} & {pct['Line']:.1f} & {pct['Arc']:.1f} & "
        f"{pct['Joint move']:.1f} & {item['distinct_families_per_program']:.2f} \\\\"
    )
    if item["author"] == "LLM" and item["robot"] != "Roomba":
        family_rows.append("\\addlinespace[1pt]")

family_table = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Robot} & \textbf{Author} & \textbf{Osc.} & \textbf{Line} & \textbf{Arc} & \textbf{Joint} & \textbf{Types/cue} \\
\midrule
""" + "\n".join(family_rows) + r"""
\bottomrule
\end{tabular}
\caption{Motion-family use in the matched 30-cue set. The four family columns report percentages among dynamic primitives. Oscillation includes repeated or signed back-and-forth \texttt{move} or \texttt{path} calls; Line and Arc contain the remaining paths; Joint contains one-way articulated moves. Types/cue is the mean number of distinct dynamic families in a program.}
\label{tab:nonanthro-motion-families}
\end{table}
"""
OUT_FAMILY_TEX.write_text(family_table)

print(OUT_JSON)
print(OUT_TEX)
print(OUT_FAMILY_TEX)
for item in summary:
    print(item)
