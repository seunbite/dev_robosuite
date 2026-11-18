#!/usr/bin/env python3
"""Plot 30-cue attribute retention across increasingly constrained bodies."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
REFERENCE_SOURCE = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
TARGET_SOURCE = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
OUT_PDF = HERE / "figure_attribute_retention_lines.pdf"
OUT_JSON = HERE / "figure_attribute_retention_data.json"

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
LABELS = ("Hum.", "Tiago", "S. arm", "Stretch", "Roomba")
COLORS = {"movement": "#0F8F83", "reach": "#7C4DCC", "palm": "#64748B"}
SYMMETRIC_SIGN_AXES = {"y", "pan", "tilt"}


def source_code(row: dict, platform: str) -> str:
    return row["platforms"][platform].get("source_code", "").lower().replace("\n", " ")


def last_reach(code: str) -> str | None:
    values = []
    aliases = {"left": "in", "right": "out", "center": "still"}
    for match in re.finditer(r"\bdir\s*=\s*([^,)]+)", code):
        value = aliases.get(match.group(1).split("|")[0].strip(), match.group(1).split("|")[0].strip())
        if value != "still":
            values.append(value)
    return values[-1] if values else None


def last_palm_hv(code: str) -> str | None:
    """Collapse detailed palm normals to the H/V gripper class."""
    values = []
    for match in re.finditer(r"\bgrip\s*=\s*([^,)]+)", code):
        value = match.group(1).replace("horizontal", "horiz").replace("vertical", "vert")
        hv = re.search(r"\b(horiz|vert)\b", value)
        if hv:
            values.append(hv.group(1))
    return values[-1] if values else None


def _is_oscillatory_value(value: str) -> bool:
    if "±" in value or "+-" in value or "-+" in value:
        return True
    numbers = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", value)]
    # A list of distinct targets (e.g., [-15, 0]) explicitly moves between
    # positions and is therefore back-and-forth. A scalar repeated with
    # ``rep`` remains a one-way directional stroke.
    if "[" in value and "]" in value and len(set(numbers)) >= 2:
        return True
    return any(x > 0 for x in numbers) and any(x < 0 for x in numbers)


def movement_class(code: str) -> str:
    """Return a carrier-agnostic motion-form class.

    This deliberately ignores which joint or body executes the stroke so that,
    for example, a head oscillation can be retained as a wrist or base rock.
    """
    operations = []
    for match in re.finditer(r"\b(move|path)\(([^)]*)\)", code):
        kind, body = match.group(1), match.group(2)
        tokens = [part.strip() for part in body.split(",")]

        if kind == "path" and "arc" in tokens:
            radius_match = re.search(r"\br\s*=\s*([0-9.]+)", body)
            radius = float(radius_match.group(1)) if radius_match else None
            angle_match = re.search(r"(?:sweep|deg)\s*=\s*([+-]?\d+(?:\.\d+)?)", body)
            angle = abs(float(angle_match.group(1))) if angle_match else None
            if (radius is None or radius > 0.001) and angle is not None and angle >= 300:
                return "circular"

        values = []
        for pattern in (
            r"\bdist\s*=\s*(\[[^\]]+\]|[^,]+)",
            r"\bdeg\s*=\s*(\[[^\]]+\]|[^,]+)",
            r"\b[xyz]\s*=\s*(\[[^\]]+\]|[^,]+)",
        ):
            values.extend(re.findall(pattern, body))
        key = None
        if kind == "move":
            positional = re.match(r"\s*([^,]+),\s*([^,]+),\s*(\[[^\]]+\]|[^,]+)", body)
            if positional and "=" not in positional.group(1):
                key = (positional.group(1).strip(), positional.group(2).strip())
                values.append(positional.group(3))
        # Repeating a one-way stroke does not make its motion form oscillatory.
        # Only an explicitly signed back-and-forth value (e.g., [-20, 5] or
        # +/-10) is treated as oscillation; ``rep`` is timing, not direction.
        signs = {
            1 if value > 0 else -1
            for token in values
            for value in [float(number) for number in re.findall(r"[+-]?\d+(?:\.\d+)?", token)]
            if value != 0
        }
        operations.append((key, any(_is_oscillatory_value(v) for v in values), signs))

    if not operations:
        return "static"
    if any(is_oscillatory for _, is_oscillatory, _ in operations):
        return "oscillatory"
    for (previous_key, _, previous_signs), (key, _, signs) in zip(operations, operations[1:]):
        if key is not None and key == previous_key and previous_signs | signs == {-1, 1}:
            return "oscillatory"
    return "directional"


def _stroke_signs(values: list[str]) -> set[str]:
    """Return signed displacement, with named height targets mapped to +/- ."""
    raw = " ".join(values).lower()
    if re.search(r"\bhigh\b", raw):
        return {"+"}
    if re.search(r"\blow\b", raw):
        return {"-"}
    signs = set()
    if "±" in raw or "+-" in raw or "-+" in raw:
        signs.update({"+", "-"})
    for number in re.findall(r"[+-]?\d+(?:\.\d+)?", raw):
        value = float(number)
        if value > 0:
            signs.add("+")
        elif value < 0:
            signs.add("-")
    return signs or {"?"}


def movement_signature(code: str) -> str:
    """Canonical movement form plus its axis/sign evidence.

    The sign of lateral y, head pan, and head tilt is deliberately collapsed:
    mirrored left/right realizations are equivalent for this alignment metric.
    """
    form = movement_class(code)
    if form == "static":
        return "static"

    features = set()
    for match in re.finditer(r"\b(move|path)\(([^)]*)\)", code):
        kind, body = match.group(1), match.group(2)
        axis = None
        values = []
        if kind == "move":
            positional = re.match(r"\s*[^,]+,\s*([^,]+),\s*(\[[^\]]+\]|[^,]+)", body)
            if positional:
                axis = positional.group(1).strip()
                values.append(positional.group(2))
        if axis is None:
            axis_match = re.search(r"\baxis\s*=\s*([^, )]+)", body)
            if axis_match:
                axis = axis_match.group(1).strip()
        if axis is None and kind == "path":
            coordinate_match = re.search(r"\b([xyz])\s*=", body)
            if coordinate_match:
                axis = coordinate_match.group(1)
            elif "arc" in body:
                plane_match = re.search(r"\bplane\s*=\s*([^, )]+)", body)
                axis = f"arc-{plane_match.group(1)}" if plane_match else "yaw"
        for pattern in (
            r"\bdist\s*=\s*(\[[^\]]+\]|[^,]+)",
            r"\bdeg\s*=\s*(\[[^\]]+\]|[^,]+)",
            r"\b[xyz]\s*=\s*(\[[^\]]+\]|[^,]+)",
        ):
            values.extend(re.findall(pattern, body))
        if axis is not None:
            signs = {"sym"} if axis in SYMMETRIC_SIGN_AXES else _stroke_signs(values)
            features.update((axis, sign) for sign in signs)

    return form + ":" + ",".join(f"{axis}{sign}" for axis, sign in sorted(features))


def retention(reference_rows: list[dict], target_rows: list[dict]) -> dict:
    targets_by_cue = {row["cue"]: row for row in target_rows}
    reference = {}
    for row in reference_rows:
        code = source_code(row, "humanoid")
        reference[row["cue"]] = {
            "movement": movement_signature(code),
            "reach": last_reach(code),
            "palm": last_palm_hv(code),
        }

    payload = {
        "n_cues": len(reference_rows),
        "reference": "human-authored humanoid",
        "target": "Gemini-2.5-Pro",
        "platforms": {},
    }
    for platform in PLATFORMS:
        cells = []
        for row in reference_rows:
            cue = row["cue"]
            target_code = source_code(targets_by_cue[cue], platform)
            ref = reference[row["cue"]]
            target = {
                "movement": movement_signature(target_code),
                "reach": None if platform == "roomba" else last_reach(target_code),
                "palm": None if platform == "roomba" else last_palm_hv(target_code),
            }
            cells.append(
                {
                    "cue": row["cue"],
                    "reference": ref,
                    "target": target,
                    "preserved": {
                        key: (
                            None
                            if platform == "roomba" and key in {"reach", "palm"}
                            else (ref[key] == target[key] if ref[key] is not None else None)
                        )
                        for key in ("movement", "reach", "palm")
                    },
                }
            )

        summary = {}
        for key in ("movement", "reach", "palm"):
            applicable = [cell["preserved"][key] for cell in cells if cell["preserved"][key] is not None]
            n_ok = sum(bool(value) for value in applicable)
            summary[key] = {
                "n": len(applicable),
                "n_ok": n_ok,
                "pct": 100.0 * n_ok / len(applicable) if applicable else None,
            }
        payload["platforms"][platform] = {"summary": summary, "cells": cells}
    return payload


def plot(payload: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 5.8,
            "axes.labelsize": 6.2,
            "legend.fontsize": 5.4,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(2.28, 2.30))
    x = list(range(len(PLATFORMS)))
    styles = {
        "movement": ("o", "Movement"),
        "reach": ("s", "Reach"),
        "palm": ("^", "Palm"),
    }
    for key in ("movement",):
        values = [
            payload["platforms"][platform]["summary"][key]["pct"]
            if payload["platforms"][platform]["summary"][key]["pct"] is not None
            else float("nan")
            for platform in PLATFORMS
        ]
        marker, label = styles[key]
        ax.plot(
            x,
            values,
            color=COLORS[key],
            marker=marker,
            markersize=3.8,
            linewidth=1.6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=3,
        )

    ax.set_xlim(-0.35, len(x) - 0.45)
    ax.set_ylim(-3, 105)
    ax.set_xticks(x, LABELS)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Retention (%)")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.tick_params(length=2, color="#94A3B8")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False, borderaxespad=0, handlelength=1.5, columnspacing=0.8)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.18, top=0.79)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main() -> None:
    reference_rows = json.loads(REFERENCE_SOURCE.read_text(encoding="utf-8"))
    target_rows = json.loads(TARGET_SOURCE.read_text(encoding="utf-8"))
    payload = retention(reference_rows, target_rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload)
    compact = {
        platform: {
            key: (
                round(payload["platforms"][platform]["summary"][key]["pct"], 1)
                if payload["platforms"][platform]["summary"][key]["pct"] is not None
                else None
            )
            for key in ("movement", "reach", "palm")
        }
        for platform in PLATFORMS
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
