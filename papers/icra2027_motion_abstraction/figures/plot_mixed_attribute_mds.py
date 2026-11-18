#!/usr/bin/env python3
"""Visualize morphology-specific Motion-Code geometry with typed MDS.

The Motion Code grammar deliberately mixes categorical pose tags, body-relative
locations, and continuous temporal parameters.  This script therefore never
concatenates raw fields into a Euclidean feature vector.  Instead it computes
one semantic distance for each of orientation (O), location (L), and movement
(M), averages the active groups, and uses classical metric MDS only as a
two-dimensional visualization of that precomputed dissimilarity matrix.

The input is the 30-cue Gemini-2.5-Pro inventory, plus an independently
generated Roomba transfer gallery. The maps are rotation-aligned to the
humanoid map without rescaling; their shared axes thus retain the relative
compression caused by a constrained channel set. Both Roomba conditions use
their documented movement-only channel set H(Roomba) = {M_base}.
"""
from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
SOURCE = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
TRANSFER_GALLERY = REPO / "data/results/roomba_strategy_fewshot/gallery30.html"
OUT_PDF = HERE / "figure_mixed_attribute_mds.pdf"
OUT_PNG = HERE / "figure_mixed_attribute_mds.png"
OUT_JSON = HERE / "figure_mixed_attribute_mds_data.json"
OUT_HTML = HERE / "figure_mixed_attribute_mds.html"

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba", "roomba_transfer")
TITLES = ("Humanoid", "Tiago", "Single arm", "Stretch", "Roomba", "Roomba - transfer")
ACTIVE_GROUPS = {
    "humanoid": ("O", "L", "M"),
    "tiago": ("O", "L", "M"),
    "single_arm": ("O", "L", "M"),
    "stretch": ("O", "L", "M"),
    "roomba": ("M",),
    "roomba_transfer": ("M",),
}
# Four equally spaced stops in a black-to-purple-to-magenta-to-yellow linear
# ramp. The ordering follows the two selected pairs.
HIGHLIGHTS = (
    ("point_you", "#0B0B0B"),
    ("hands_up_surrender", "#6B2A8C"),
    ("stretch_arms_up", "#C13A8A"),
    ("facepalm", "#F4E56A"),
)
BACKGROUND_COLOR = "#CBD2D9"
TRANSFER_BACKGROUND_COLOR = "#7F8994"
SYMMETRIC_SIGN_AXES = {"y", "pan", "tilt"}


def split_args(body: str) -> list[str]:
    """Split function arguments while preserving comma-separated lists."""
    parts, current, depth = [], [], 0
    for char in body:
        if char in "[({":
            depth += 1
        elif char in "]) }":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def params(body: str) -> dict[str, str]:
    result = {}
    for item in split_args(body):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip().lower()
    return result


def canonical(value: str) -> str:
    """Merge harmless spelling variants, retaining semantic categories."""
    value = value.lower().strip()
    value = value.replace("vertical", "vert").replace("horizontal", "horiz")
    value = value.replace("center", "centre")
    return value


def numbers(value: str) -> list[float]:
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", value)]


def representative_number(value: str) -> float | None:
    values = numbers(value)
    return float(np.mean(np.abs(values))) if values else None


def is_oscillatory(value: str) -> bool:
    if "±" in value or "+-" in value or "-+" in value:
        return True
    vals = numbers(value)
    return "[" in value and len(set(vals)) >= 2 or (any(v > 0 for v in vals) and any(v < 0 for v in vals))


def signs(value: str, axis: str) -> set[str]:
    if axis in SYMMETRIC_SIGN_AXES:
        return {"sym"}
    if "±" in value or "+-" in value or "-+" in value:
        return {"+", "-"}
    values = numbers(value)
    result = {"+" if x > 0 else "-" for x in values if x != 0}
    return result or {"?"}


def parse_code(source: str) -> dict:
    """Return typed O/L/M descriptors from one compact Motion Code program."""
    code = source.lower().replace("\n", " ")
    pose = {}
    for match in re.finditer(r"\bpose\(([^)]*)\)", code):
        pose.update(params(match.group(1)))

    orientation = {
        key: canonical(value)
        for key, value in pose.items()
        if key in {"dir", "grip", "head", "torso", "elbow", "left_arm"}
    }
    location = {}
    for key in ("x", "y", "z"):
        if key in pose:
            value = representative_number(pose[key])
            if value is not None:
                location[key] = value

    operations = []
    for match in re.finditer(r"\b(move|path)\(([^)]*)\)", code):
        kind, body = match.group(1), match.group(2)
        parts = split_args(body)
        kw = params(body)
        carrier = canonical(parts[0]) if parts else "?"
        primitive = canonical(parts[1]) if len(parts) > 1 and "=" not in parts[1] else kind
        axis = canonical(kw["axis"]) if "axis" in kw else None
        value = None
        if kind == "move" and len(parts) > 2 and "=" not in parts[1]:
            axis = canonical(parts[1])
            value = parts[2]
        elif "dist" in kw:
            value = kw["dist"]
        elif "len" in kw:
            value = kw["len"]
        elif "d" in kw:
            value = kw["d"]
        elif "deg" in kw:
            axis = axis or ("yaw" if primitive == "arc" else None)
            value = kw["deg"]
        elif "sweep" in kw:
            axis = axis or f"arc-{canonical(kw.get('plane', 'unknown'))}"
            value = kw["sweep"]
        else:
            for coordinate in ("x", "y", "z"):
                if coordinate in kw:
                    axis, value = coordinate, kw[coordinate]
                    break
        if primitive == "arc" and axis is None:
            axis = f"arc-{canonical(kw.get('plane', 'yaw'))}"
        operations.append(
            {
                "carrier": carrier,
                "primitive": primitive,
                "axis": axis or "unknown",
                "value": value or "",
                "rep": representative_number(kw["rep"]) if "rep" in kw else 1.0,
                "spd": representative_number(kw["spd"]) if "spd" in kw else None,
                "extent_key": next((key for key in ("dist", "len", "d", "deg", "sweep") if key in kw), None),
                "extent": representative_number(value or ""),
            }
        )

    if not operations:
        form = "static"
    elif any(
        op["primitive"] == "arc" and op["extent_key"] in {"deg", "sweep"} and (op["extent"] or 0) >= 300
        for op in operations
    ):
        form = "circular"
    elif any(is_oscillatory(op["value"]) for op in operations):
        form = "oscillatory"
    else:
        form = "directional"

    axes = {op["axis"] for op in operations}
    direction = {(op["axis"], sign) for op in operations for sign in signs(op["value"], op["axis"])}
    carriers = {op["carrier"] for op in operations}
    reps = [op["rep"] for op in operations]
    speeds = [op["spd"] for op in operations if op["spd"] is not None]
    extents = [(op["extent_key"], op["extent"]) for op in operations if op["extent_key"] and op["extent"] is not None]
    return {
        "O": orientation,
        "L": location,
        "M": {
            "form": form,
            "axes": axes,
            "direction": direction,
            "carriers": carriers,
            "rep": float(np.mean(reps)) if reps else None,
            "speed": float(np.mean(speeds)) if speeds else None,
            "extent": extents,
        },
    }


def categorical_distance(a: str | None, b: str | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    return float(a != b)


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


def normalized_distance(a: float | None, b: float | None, scale: float) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    return min(1.0, abs(a - b) / scale)


def orientation_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return float(np.mean([categorical_distance(a.get(key), b.get(key)) for key in keys])) if keys else 0.0


def location_distance(a: dict, b: dict) -> float:
    keys = ("x", "y", "z")
    terms = [normalized_distance(a.get(key), b.get(key), 100.0) for key in keys]
    return float(np.mean(terms))


def movement_distance(a: dict, b: dict) -> float:
    """Semantically weighted Gower-style distance inside the movement group."""
    form_distance = categorical_distance(a["form"], b["form"])
    if a["form"] == b["form"] == "static":
        return form_distance
    ext_a = {kind: value for kind, value in a["extent"]}
    ext_b = {kind: value for kind, value in b["extent"]}
    extent_keys = set(ext_a) | set(ext_b)
    extent_distance = (
        float(np.mean([normalized_distance(ext_a.get(key), ext_b.get(key), 360.0 if key in {"deg", "sweep"} else 0.5) for key in extent_keys]))
        if extent_keys
        else 0.0
    )
    terms = (
        (0.30, form_distance),
        (0.22, jaccard_distance(a["axes"], b["axes"])),
        (0.16, jaccard_distance(a["direction"], b["direction"])),
        (0.12, jaccard_distance(a["carriers"], b["carriers"])),
        (0.09, normalized_distance(a["rep"], b["rep"], 4.0)),
        (0.07, normalized_distance(a["speed"], b["speed"], 4.0)),
        (0.04, extent_distance),
    )
    return sum(weight * distance for weight, distance in terms)


def mixed_distance(a: dict, b: dict, groups: tuple[str, ...]) -> float:
    group_values = {
        "O": orientation_distance(a["O"], b["O"]),
        "L": location_distance(a["L"], b["L"]),
        "M": movement_distance(a["M"], b["M"]),
    }
    return float(np.mean([group_values[group] for group in groups]))


def classical_mds(distance: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    coords = eigvec[:, :2] * np.sqrt(np.maximum(eigval[:2], 0.0))
    embedded = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(distance, dtype=bool), 1)
    stress = float(np.sqrt(np.mean((embedded[mask] - distance[mask]) ** 2)))
    return coords, stress


def rotate_to_reference(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply only an orthogonal transform; preserve each map's MDS scale."""
    u, _, vt = np.linalg.svd(coords.T @ reference)
    return coords @ (u @ vt)


def normalize_cue(value: str) -> str:
    return re.sub(r"\s+", "_", unescape(value).strip().lower())


def load_transfer_codes(cues: list[str]) -> list[str]:
    """Extract cue/program pairs from the strategy-conditioned Roomba gallery."""
    gallery = TRANSFER_GALLERY.read_text(errors="replace")
    matches = re.findall(r'<div class="cue">\s*\d+\s*·\s*([^<]+)</div><code>(.*?)</code>', gallery, flags=re.DOTALL)
    code_by_cue = {normalize_cue(cue): unescape(code).strip() for cue, code in matches}
    missing = [cue for cue in cues if cue not in code_by_cue]
    if missing:
        raise ValueError(f"Transfer gallery is missing cues: {', '.join(missing)}")
    return [code_by_cue[cue] for cue in cues]


def build_payload(rows: list[dict]) -> dict:
    cues = [row["cue"] for row in rows]
    categories = [row["category"] for row in rows]
    transfer_codes = load_transfer_codes(cues)
    result = {
        "source": str(SOURCE.relative_to(REPO)),
        "transfer_source": str(TRANSFER_GALLERY.relative_to(REPO)),
        "n_cues": len(rows),
        "projection": "classical metric MDS over a precomputed typed mixed-attribute distance",
        "group_weights": {"O": 1 / 3, "L": 1 / 3, "M": 1 / 3},
        "platforms": {},
    }
    raw_coords = {}
    for platform in PLATFORMS:
        source_codes = transfer_codes if platform == "roomba_transfer" else [row["platforms"][platform].get("source_code", "") for row in rows]
        descriptors = [parse_code(code) for code in source_codes]
        matrix = np.zeros((len(rows), len(rows)))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                matrix[i, j] = matrix[j, i] = mixed_distance(descriptors[i], descriptors[j], ACTIVE_GROUPS[platform])
        coords, stress = classical_mds(matrix)
        raw_coords[platform] = coords
        result["platforms"][platform] = {
            "active_groups": list(ACTIVE_GROUPS[platform]),
            "stress_rmse": stress,
            "mean_pairwise_distance": float(matrix[np.triu_indices(len(rows), 1)].mean()),
            "distance_matrix": matrix.round(6).tolist(),
            "descriptors": descriptors,
            "source_codes": source_codes,
        }

    reference = raw_coords["humanoid"]
    for platform in PLATFORMS:
        coords = raw_coords[platform] if platform == "humanoid" else rotate_to_reference(raw_coords[platform], reference)
        result["platforms"][platform]["coordinates"] = coords.round(6).tolist()
    result["cues"] = cues
    result["categories"] = categories
    return result


def plot(payload: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.7,
            "axes.titlesize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    all_coords = np.vstack([np.asarray(payload["platforms"][platform]["coordinates"]) for platform in PLATFORMS])
    limit = 1.13 * np.max(np.abs(all_coords))
    fig, axes = plt.subplots(1, len(PLATFORMS), figsize=(7.16, 1.90), sharex=True, sharey=True)
    cues = payload["cues"]
    for ax, platform, title in zip(axes, PLATFORMS, TITLES):
        coords = np.asarray(payload["platforms"][platform]["coordinates"])
        background_color = TRANSFER_BACKGROUND_COLOR if platform == "roomba_transfer" else BACKGROUND_COLOR
        ax.scatter(coords[:, 0], coords[:, 1], s=13, color=background_color, edgecolor="white", linewidth=0.35, zorder=2)
        for cue, color in HIGHLIGHTS:
            idx = cues.index(cue)
            ax.scatter(coords[idx, 0], coords[idx, 1], s=22, color=color, edgecolor="#38424D", linewidth=0.40, zorder=4)
        ax.axhline(0, color="#D7DCE1", linewidth=0.42, zorder=0)
        ax.axvline(0, color="#D7DCE1", linewidth=0.42, zorder=0)
        ax.set_title(title, pad=4.0, color="#202833")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#C9D0D7")
            spine.set_linewidth(0.45)
        ax.set_box_aspect(1)

    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="#38424D", markeredgewidth=0.45, markersize=4.5, label=cue.replace("_", " ")) for cue, color in HIGHLIGHTS]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.120), frameon=False, fontsize=7.2, handletextpad=0.35, columnspacing=1.25)
    fig.subplots_adjust(left=0.020, right=0.998, top=0.825, bottom=0.245, wspace=0.14)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_interactive_html(payload: dict) -> None:
    """Write a self-contained Plotly explorer with hover-first cue inspection."""
    titles = {"humanoid": "Humanoid", "tiago": "Tiago", "single_arm": "Single arm", "stretch": "Stretch", "roomba": "Roomba", "roomba_transfer": "Roomba - transfer"}
    fig = make_subplots(rows=1, cols=len(PLATFORMS), subplot_titles=[titles[platform] for platform in PLATFORMS], horizontal_spacing=0.045)
    all_coords = np.vstack([np.asarray(payload["platforms"][platform]["coordinates"]) for platform in PLATFORMS])
    limit = 1.13 * np.max(np.abs(all_coords))
    cues = payload["cues"]

    for col, platform in enumerate(PLATFORMS, start=1):
        item = payload["platforms"][platform]
        coords = np.asarray(item["coordinates"])
        background_color = TRANSFER_BACKGROUND_COLOR if platform == "roomba_transfer" else BACKGROUND_COLOR
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0], y=coords[:, 1], mode="markers", showlegend=False,
                marker={"size": 8, "color": background_color, "line": {"color": "white", "width": 0.8}},
                customdata=[cue.replace("_", " ") for cue in cues], hovertemplate="<b>%{customdata}</b><extra></extra>",
            ),
            row=1, col=col,
        )
        for cue, color in HIGHLIGHTS:
            idx = cues.index(cue)
            fig.add_trace(
                go.Scatter(
                    x=[coords[idx, 0]], y=[coords[idx, 1]], mode="markers", name=cue.replace("_", " "),
                    legendgroup=cue, showlegend=col == 1,
                    marker={"size": 11, "color": color, "line": {"color": "#38424D", "width": 0.8}},
                    customdata=[cue.replace("_", " ")], hovertemplate="<b>%{customdata}</b><extra></extra>",
                ),
                row=1, col=col,
            )
        fig.update_xaxes(range=[-limit, limit], showgrid=True, gridcolor="#E0E5EA", zeroline=True, zerolinecolor="#CDD4DB", showticklabels=False, constrain="domain", row=1, col=col)
        fig.update_yaxes(range=[-limit, limit], showgrid=True, gridcolor="#E0E5EA", zeroline=True, zerolinecolor="#CDD4DB", showticklabels=False, scaleanchor=f"x{'' if col == 1 else col}", scaleratio=1, constrain="domain", row=1, col=col)

    fig.update_layout(
        title={"text": "Typed Motion-Code geometry across embodiments", "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=330,
        margin={"l": 20, "r": 18, "t": 65, "b": 78},
        hovermode="closest",
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Arial"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.01, "xanchor": "center", "x": 0.5, "font": {"size": 13}},
    )
    fig.update_annotations(font={"size": 13, "color": "#202833"})
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True, config={"responsive": True, "displayModeBar": False})


def main() -> None:
    rows = json.loads(SOURCE.read_text())
    payload = build_payload(rows)
    # Sets make the parsed categorical descriptors compact while computing;
    # serialize them deterministically for the accompanying inspection record.
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=lambda value: sorted(value) if isinstance(value, set) else value) + "\n")
    plot(payload)
    write_interactive_html(payload)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_HTML}")
    for platform in PLATFORMS:
        item = payload["platforms"][platform]
        print(f"{platform:10s} distance={item['mean_pairwise_distance']:.3f}, MDS RMSE={item['stress_rmse']:.3f}")


if __name__ == "__main__":
    main()
