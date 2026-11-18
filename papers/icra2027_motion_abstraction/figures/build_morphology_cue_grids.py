#!/usr/bin/env python3
"""Build 3×6 human-like and 5×6 non-human-like grids with Fig.8-style cell captions."""
from __future__ import annotations

import sys
import html
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
FIGURES = PAPER / "figures"
ODDKIN = PAPER / "synthetic_morphology/renders/cues30"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
sys.path.insert(0, str(REPO / "adhoc/vlm_test"))
sys.path.insert(0, str(FIGURES))

from build_analysis_qual_grids import gif_path, load_gif_frames  # noqa: E402
from build_fig1_gaze_rub_grid import (  # noqa: E402
    fill_cell,
    peak_pose_with_traj,
    track_from_first,
)
from testset_utils import _image_with_ee_path  # noqa: E402

CELL = 188
LABEL_W = 76
N_PER = 4
PAIR_GAP = 18
ROW_GAP = 8
PAD = 5
TEXT_SCALE = 1.5
FONT_SIZE = round(12 * TEXT_SCALE)
CUE_RGB = (31, 111, 235)
CAP_RGB = (30, 30, 30)
MOV_ALPHA = 0.50

CUES = {
    1: "nod_yes",
    2: "shake_head_no",
    3: "draw_circle_repeat",
    4: "gaze_shift_avoid",
    5: "fist_pump",
    8: "rub_eye_tired",
    9: "point_you",
    11: "beckon_come_here",
    12: "stop_palm_out",
    13: "salute",
    15: "highfive_invite",
    16: "shrug_idk",
    19: "arms_open_welcome",
    20: "hands_up_surrender",
    21: "stretch_arms_up",
    22: "raise_fists_victory",
    23: "cross_arms_defensive",
    25: "facepalm",
    26: "depict_tree",
    27: "depict_big_bowl",
    28: "depict_lion",
    30: "depict_flying_bird",
    31: "self_scratch_head_confused",
    42: "throat_cut_stop",
    43: "visor_search",
    44: "yield_turn",
    65: "continue_on_forward_roll",
    69: "interrupt_hard",
    70: "interrupt_soft",
    90: "clap_applause",
}

EXTENSION_HTML = REPO / "data/results/html/embodiment_extension20_gemini.html"

# Four tiles per robot; columns are not aligned across robots.
# Prefer visually distinct, well-executed realizations.
ROW_CUES: dict[str, list[int]] = {
    "humanoid": [21, 20, 16, 90],
    "tiago": [21, 20, 31, 13],
    "single_arm": [43, 8, 23, 28],
    "stretch": [69, 70, 27, 25],
    "roomba": [22, 42, 43, 65],
    "gcr_alpha1": [25, 19, 23, 11],
    "gcr_alpha0": [19, 44, 2, 5],
    "gcr_alpha05": [19, 9, 30, 16],
}

# Fig. 2 uses a balanced source split on the five catalog robots with completed
# human yes/no screening.  Each 2x2 block contains two human-authored reference
# programs (top) and two Gemini-authored programs judged identifiable (bottom).
# LLM selections are restricted to final_human_yesno=YES in
# data/results/evaluation/embodiment_cue_id/all_methods_human_yesno_c1_30.csv.
FIG2_MIXED_CUES: dict[str, list[tuple[int, str]]] = {
    "humanoid": [(25, "human"), (13, "human"), (1, "llm"), (30, "llm")],
    "tiago": [(12, "human"), (11, "human"), (2, "llm"), (19, "llm")],
    "single_arm": [(15, "human"), (30, "human"), (1, "llm"), (9, "llm")],
    "stretch": [(8, "human"), (19, "human"), (1, "llm"), (22, "llm")],
    "roomba": [(11, "human"), (12, "human"), (1, "llm"), (3, "llm")],
}

FIG2_ROBOTS = [
    ("humanoid", "Unitree G1"),
    ("tiago", "Tiago"),
    ("single_arm", "IIWA"),
    ("stretch", "Stretch"),
    ("roomba", "Roomba"),
]

# Two robots per row; names sit at the left of each group of four.
PAIR_ROWS = [
    (("humanoid", "Unitree G1"), ("tiago", "Tiago")),
    (("single_arm", "IIWA"), ("stretch", "Stretch")),
    (("roomba", "Roomba"), ("gcr_alpha1", "Paddle Claw")),
    (("gcr_alpha0", "Trileg Crown"), ("gcr_alpha05", "Spine Crab")),
]

# Fig.8-style one-liners: what the motion looks like on that tile.
CELL_CAPS = {
    ("single_arm", 2): "Gripper yaws left-right.",
    ("stretch", 27): "Gripper traces a bowl rim.",
    ("stretch", 22): "Arm and mast rise in victory.",
    ("roomba", 22): "Base spins twice (720 degrees).",
    ("roomba", 3): "Base traces repeated circles.",
    # --- Unitree G1 ---
    ("humanoid", 1): "Head nods down-up, three times.",
    ("humanoid", 4): "Head turns away and holds.",
    ("humanoid", 5): "Raise a fist, then one sharp pump.",
    ("humanoid", 11): "Palm faces you, then pulls in repeatedly.",
    ("humanoid", 19): "Both arms open wide and hold.",
    ("humanoid", 30): "Arms out, then flap like wings.",
    ("humanoid", 13): "Fingertips at the brow, then hold.",
    ("humanoid", 25): "Palm covers the face, head drops.",
    ("humanoid", 16): "Both arms open outward in a shrug.",
    ("humanoid", 20): "Both hands rise with palms outward.",
    ("humanoid", 21): "Both arms extend overhead and hold.",
    ("humanoid", 90): "Both hands meet repeatedly to clap.",
    # --- Tiago ---
    ("tiago", 1): "Head nods down-up, three times.",
    ("tiago", 4): "Head turns away, then the base spins.",
    ("tiago", 5): "Raise the arm, then one sharp pump.",
    ("tiago", 11): "Hand fronts you, then draws in repeatedly.",
    ("tiago", 19): "Both arms open wide and hold.",
    ("tiago", 30): "Arms out, then flap like wings.",
    ("tiago", 2): "Head turns left-right, three times.",
    ("tiago", 12): "Palm fronts you and holds.",
    ("tiago", 15): "Arm rises, then presents a highfive.",
    ("tiago", 13): "One gripper rises to the forehead.",
    ("tiago", 20): "Both grippers rise in surrender.",
    ("tiago", 21): "Both grippers rise overhead and hold.",
    ("tiago", 31): "One gripper scratches near the head.",
    # --- Gesture Tree (α=1) ---
    ("gcr_alpha1", 1): "Head nods down-up, three times.",
    ("gcr_alpha1", 4): "Looks left, then right, then front.",
    ("gcr_alpha1", 5): "Raise the claw, then pump it up.",
    ("gcr_alpha1", 11): "Open claw fronts you, then pulls in.",
    ("gcr_alpha1", 19): "Paddle and claw open wide and hold.",
    ("gcr_alpha1", 30): "Both branches flap side to side.",
    ("gcr_alpha1", 15): "Claw rises high, then holds.",
    ("gcr_alpha1", 25): "Paddle covers the face, head drops.",
    # --- Single Arm ---
    ("single_arm", 1): "Shoulder rocks up-down, three times.",
    ("single_arm", 4): "Wrist turns away and holds.",
    ("single_arm", 5): "Arm rises, then one sharp pump.",
    ("single_arm", 11): "Hand fronts you, then draws in.",
    ("single_arm", 19): "Arm opens outward and holds.",
    ("single_arm", 30): "Arm out, then flaps up and down.",
    ("single_arm", 9): "Gripper points at you and holds.",
    ("single_arm", 15): "Arm rises, then presents a highfive.",
    ("single_arm", 8): "Gripper rubs near the camera.",
    ("single_arm", 23): "Arm folds inward defensively.",
    ("single_arm", 28): "Arm advances in a lion-like strike.",
    ("single_arm", 43): "Gripper shades and scans ahead.",
    # --- Stretch ---
    ("stretch", 1): "Camera nods down-up twice.",
    ("stretch", 4): "Camera pans while the base turns.",
    ("stretch", 5): "Reach high, then pump once.",
    ("stretch", 11): "Roll back a bit, then pull in.",
    ("stretch", 19): "Reach up while the base spins open.",
    ("stretch", 30): "Arm flaps while the base spins around.",
    ("stretch", 8): "Gripper rubs near the camera.",
    ("stretch", 12): "Arm fronts you, then holds.",
    ("stretch", 25): "Gripper covers the camera face.",
    ("stretch", 27): "Gripper traces a bowl rim.",
    ("stretch", 69): "Arm cuts in sharply to interrupt.",
    ("stretch", 70): "Arm enters gently to interrupt.",
    # --- Roomba ---
    ("roomba", 1): "Rocks forward-back, three times.",
    ("roomba", 4): "Spins in place away, then holds.",
    ("roomba", 5): "One sharp surge forward, then holds.",
    ("roomba", 11): "Retreats, turns, then little wags.",
    ("roomba", 19): "Advances, then sweeps left and right.",
    ("roomba", 30): "Big circle, surge, then circle again.",
    ("roomba", 2): "Spins left-right in place, three times.",
    ("roomba", 12): "Surges forward, then holds.",
    ("roomba", 26): "Traces a tall vertical outline.",
    ("roomba", 22): "Base spins twice in victory.",
    ("roomba", 42): "Base sweeps sharply across its front.",
    ("roomba", 43): "Base scans side to side.",
    ("roomba", 65): "Base rolls forward repeatedly.",
    # --- Trileg Crown (α=0) ---
    ("gcr_alpha0", 1): "Base wiggles yaw, three tiny nods.",
    ("gcr_alpha0", 4): "Faces left, then right, then front.",
    ("gcr_alpha0", 5): "Raises one leg, then pumps the hip.",
    ("gcr_alpha0", 11): "Raises a leg, then pulls the knee in.",
    ("gcr_alpha0", 19): "Raises all three legs and holds.",
    ("gcr_alpha0", 30): "Two hips flap side to side.",
    ("gcr_alpha0", 2): "Base yaws left-right, three times.",
    ("gcr_alpha0", 12): "Raises a leg toward you, then holds.",
    ("gcr_alpha0", 15): "Raises a leg high, then pumps.",
    ("gcr_alpha0", 44): "Front leg yields and opens the turn.",
    # --- Spine Crab (α=0.5) ---
    ("gcr_alpha05", 1): "Spine tips down-up, three times.",
    ("gcr_alpha05", 4): "Body turns left, right, then front.",
    ("gcr_alpha05", 5): "Raises the long pincer, then pumps.",
    ("gcr_alpha05", 11): "Long tip fronts you, then draws in.",
    ("gcr_alpha05", 19): "Both pincers open wide and hold.",
    ("gcr_alpha05", 30): "Both pincers flap side to side.",
    ("gcr_alpha05", 9): "Long pincer points at you and holds.",
    ("gcr_alpha05", 25): "Long pincer covers the marker plate.",
    ("gcr_alpha05", 16): "Both pincers open in a shrug.",
    ("gcr_alpha1", 23): "Both branches fold inward defensively.",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, box, text: str, font, fill="#202020") -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=1, align="center")
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=1,
        align="center",
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                cur = ""
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines or [""]


def _chip(
    base: Image.Image,
    xy: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int], ImageFont.ImageFont]],
    *,
    pad: int = 5,
) -> Image.Image:
    draw = ImageDraw.Draw(base)
    line_h = FONT_SIZE + 4
    tw = max(int(draw.textlength(text, font=font)) for text, _, font in lines)
    th = line_h * len(lines)
    x, y = xy
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(
        [x, y, x + tw + 2 * pad, y + th + 2 * pad],
        radius=5,
        fill=(255, 255, 255, 250),
        outline=(200, 200, 200, 255),
        width=1,
    )
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    d = ImageDraw.Draw(out)
    cy = y + pad
    for text, color, font in lines:
        d.text((x + pad, cy), text, fill=color, font=font)
        cy += line_h
    return out.convert("RGB")


def _draw_cue_name(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    platform: str,
) -> Image.Image:
    x, y = xy
    if platform == "roomba":
        return _chip(canvas, xy, [(text, (20, 20, 20), font)], pad=4)
    color = (255, 255, 255) if platform.startswith("gcr_") else (20, 20, 20)
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), text, fill=color, font=font)
    return canvas


@lru_cache(maxsize=None)
def _html_llm_gif(platform: str, idx: int) -> Path | None:
    """Resolve the exact GIF shown in the extension gallery.

    Some extension cues use a local render index (for example, cue 31 is
    ``emb_g01`` in the pilot75 directory), so reconstructing the filename from
    the global cue ID can silently select a different render.
    """
    if not EXTENSION_HTML.is_file():
        return None
    source = EXTENSION_HTML.read_text(encoding="utf-8", errors="replace")
    heading = re.search(
        rf'<h2>ID\s+{idx}\s+·\s+{re.escape(CUES[idx])}</h2>', source
    )
    if heading is None:
        return None
    section_end = source.find('<section class="cue"', heading.end())
    section = source[heading.start(): section_end if section_end >= 0 else len(source)]
    article = re.search(
        rf'<article[^>]*data-platform="{re.escape(platform)}".*?</article>',
        section,
        re.DOTALL,
    )
    if article is None:
        return None
    image = re.search(r'<img[^>]+src="([^"]+)"', article.group(0))
    if image is None:
        return None
    src = html.unescape(image.group(1)).split("?", 1)[0]
    path = (EXTENSION_HTML.parent / src).resolve()
    return path if path.is_file() else None


def _catalog_gif(platform: str, idx: int, source: str = "human") -> Path | None:
    if source == "llm":
        gallery_path = _html_llm_gif(platform, idx)
        if gallery_path is not None:
            return gallery_path
        llm_dirs = {
            "humanoid": "humanoid/gif",
            "tiago": "tiago/gif",
            "single_arm": "manipulator/gif/IIWA",
            "stretch": "stretch/gif",
            "roomba": "roomba/gif",
        }
        path = (REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
                / llm_dirs[platform] / f"emb_g{idx:02d}_{CUES[idx]}.gif")
        if path.is_file():
            return path
        filename = f"emb_g{idx:02d}_{CUES[idx]}.gif"
        extension_candidates = [
            REPO / "data/results/render/embodiment_extension20_gemini" / llm_dirs[platform] / filename,
            REPO / "data/results/render/embodiment_motion_code_synth60_motion60" / platform / "gif" / filename,
            REPO / "data/results/render/embodiment_motion_code_synth60_tiago60" / platform / "gif" / filename,
        ]
        for candidate in extension_candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(path)
    path = gif_path("gt", platform, idx, CUES, {})
    if path is not None and path.is_file():
        return path
    filename = f"emb_g{idx:02d}_{CUES[idx]}.gif"
    extension_candidates = [
        REPO / "data/results/render/embodiment_motion_code_synth60_motion60" / platform / "gif" / filename,
        REPO / "data/results/render/embodiment_motion_code_synth60_tiago60" / platform / "gif" / filename,
    ]
    for candidate in extension_candidates:
        if candidate.is_file():
            return candidate
    return None


def _oddkin_gif(platform: str, idx: int) -> Path:
    filename = f"{CUES[idx]}.gif"
    candidates = [
        PAPER / "synthetic_morphology/renders/cues50" / platform / filename,
        ODDKIN / platform / filename,
        REPO / "data/results/render/embodiment_motion_code_paddlebot90" / platform / "gif" / f"emb_g{idx:02d}_{filename}",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _oddkin_frames(path: Path) -> list[np.ndarray]:
    raw = load_gif_frames(path)
    frames: list[np.ndarray] = []
    for frame in raw:
        height = frame.shape[0]
        top = int(0.12 * height)
        bottom = int(0.10 * height)
        frames.append(frame[top : height - bottom])
    return frames


def _oddkin_cell(path: Path, idx: int, *, trajectory: bool) -> Image.Image:
    frames = _oddkin_frames(path)
    pose_idx = min(len(frames) - 1, max(0, int(0.58 * (len(frames) - 1))))
    base = Image.fromarray(frames[pose_idx]).convert("RGB")
    if trajectory:
        points = track_from_first(frames, "oddkin", idx=idx)
        if len(points) >= 2:
            base = _image_with_ee_path(base, points)
    return fill_cell(base, CELL, platform="oddkin")


def _g1_cover(img: Image.Image, size: int) -> Image.Image:
    """Fig.~2 G1 crop: square cover, slight upward bias, no sky pad."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = max(0, (h - side) // 2 - int(round(0.04 * side)))
    y0 = min(y0, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def _humanoid_fig2_cell(path: Path, idx: int, *, show_trajectory: bool) -> Image.Image:
    """Opaque peak pose + robot-only afterimage + trajectory (Fig.~2 movement cell)."""
    from build_analysis_qual_grids import load_gif_frames
    from build_attr_buildup_grid import _draw_traj_on, _overlay_afterimage

    raw = load_gif_frames(path)
    frames = [fr[:-14] for fr in raw] if raw else []
    if not frames:
        missing = Image.new("RGB", (CELL, CELL), "#f2f2f2")
        return missing
    n = len(frames) - 1
    i_ghost = max(0, int(0.22 * n))
    i_loc = max(i_ghost + 1, int(0.62 * n))
    loc = _g1_cover(Image.fromarray(frames[i_loc]), CELL)
    ghost = _g1_cover(Image.fromarray(frames[i_ghost]), CELL)
    stacked = _overlay_afterimage(loc, ghost, MOV_ALPHA, platform="humanoid")
    step = max(1, len(frames) // 16)
    cell_frames = [
        np.asarray(_g1_cover(Image.fromarray(fr), CELL)) for fr in frames[::step]
    ]
    if show_trajectory:
        pts = track_from_first(cell_frames, "humanoid", idx=idx)
        return _draw_traj_on(stacked, pts)
    return stacked


def _cell(platform: str, idx: int, *, trajectory: bool, source: str = "human") -> Image.Image:
    if platform.startswith("gcr_"):
        path = _oddkin_gif(platform, idx)
        if path.is_file():
            return _oddkin_cell(path, idx, trajectory=trajectory)
    else:
        path = _catalog_gif(platform, idx, source)
        if path is not None and path.is_file():
            if platform == "humanoid":
                return _humanoid_fig2_cell(path, idx, show_trajectory=False)
            image = peak_pose_with_traj(path, platform, idx)
            return fill_cell(image, CELL, platform=platform)

    missing = Image.new("RGB", (CELL, CELL), "#f2f2f2")
    _centered(ImageDraw.Draw(missing), (0, 0, CELL, CELL), "missing", _font(16))
    print(f"missing: {platform} {idx} {CUES[idx]}")
    return missing


def _cue_label(cue: str) -> str:
    short = {
        "nod_yes": "nod yes",
        "shake_head_no": "shake no",
        "gaze_shift_avoid": "gaze avoid",
        "fist_pump": "fist pump",
        "rub_eye_tired": "rub eye",
        "point_you": "point you",
        "beckon_come_here": "beckon",
        "stop_palm_out": "stop",
        "salute": "salute",
        "highfive_invite": "highfive",
        "arms_open_welcome": "open welcome",
        "facepalm": "facepalm",
        "depict_tree": "depict tree",
        "depict_big_bowl": "big bowl",
        "raise_fists_victory": "fists victory",
        "depict_flying_bird": "flying bird",
        "hands_up_surrender": "hands up",
        "self_scratch_head_confused": "scratch head",
        "cross_arms_defensive": "cross arms",
        "continue_on_forward_roll": "continue forward",
        "interrupt_hard": "hard interrupt",
        "interrupt_soft": "soft interrupt",
    }
    return short.get(cue, cue.replace("_", " "))


def _paste_robot(
    canvas: Image.Image,
    *,
    platform: str,
    label: str,
    x_label: int,
    y0: int,
    trajectory: bool,
    source: str = "human",
) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    row_font = _font(round(13 * TEXT_SCALE), bold=True)
    cue_font = _font(round(11 * TEXT_SCALE), bold=True)
    cap_font = _font(FONT_SIZE, bold=False)
    _centered(
        draw,
        (x_label, y0, x_label + LABEL_W - 4, y0 + CELL),
        label.replace(" ", "\n"),
        row_font,
    )
    idxs = ROW_CUES[platform]
    if len(idxs) != N_PER:
        raise ValueError(f"{platform}: expected {N_PER} cues, got {len(idxs)}")
    x_cells = x_label + LABEL_W
    for col, idx in enumerate(idxs):
        x0 = x_cells + col * CELL
        canvas.paste(_cell(platform, idx, trajectory=trajectory, source=source), (x0, y0))
        canvas = _draw_cue_name(
            canvas,
            (x0 + PAD, y0 + PAD),
            _cue_label(CUES[idx]),
            cue_font,
            platform=platform,
        )
        probe = ImageDraw.Draw(canvas)
        key = (platform, idx)
        if key not in CELL_CAPS:
            raise KeyError(f"missing caption for {key} ({CUES[idx]})")
        wrapped = _wrap(probe, CELL_CAPS[key], cap_font, CELL - 2 * PAD - 10, max_lines=3)
        lines = [(ln, CAP_RGB, cap_font) for ln in wrapped]
        tw = max(int(probe.textlength(t, font=f)) for t, _, f in lines)
        th = (FONT_SIZE + 4) * len(lines)
        chip_w = tw + 2 * PAD
        chip_h = th + 2 * PAD
        cx = x0 + (CELL - chip_w) // 2
        cy = y0 + CELL - chip_h - PAD
        canvas = _chip(canvas, (cx, cy), lines)
    return canvas


def build_paired(out_name: str, *, trajectory: bool) -> None:
    group_w = LABEL_W + N_PER * CELL
    width = group_w + PAIR_GAP + group_w
    height = len(PAIR_ROWS) * CELL + (len(PAIR_ROWS) - 1) * ROW_GAP
    canvas = Image.new("RGB", (width, height), "white")
    for row, (left, right) in enumerate(PAIR_ROWS):
        y0 = row * (CELL + ROW_GAP)
        canvas = _paste_robot(
            canvas,
            platform=left[0],
            label=left[1],
            x_label=0,
            y0=y0,
            trajectory=trajectory,
            source="llm" if not left[0].startswith("gcr_") else "human",
        )
        canvas = _paste_robot(
            canvas,
            platform=right[0],
            label=right[1],
            x_label=group_w + PAIR_GAP,
            y0=y0,
            trajectory=trajectory,
            source="llm" if not right[0].startswith("gcr_") else "human",
        )
        draw = ImageDraw.Draw(canvas)
        mid = group_w + PAIR_GAP // 2
        draw.line((mid, y0 + 8, mid, y0 + CELL - 8), fill="#d8d8d8", width=1)
        if row < len(PAIR_ROWS) - 1:
            draw.line(
                (0, y0 + CELL + ROW_GAP // 2, width, y0 + CELL + ROW_GAP // 2),
                fill="#d0d0d0",
            )
    out = FIGURES / out_name
    canvas.save(out, optimize=True)
    print("wrote", out, canvas.size)


def _source_badge(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    source: str,
) -> Image.Image:
    label = "LLM ✓" if source == "llm" else "Human"
    color = (25, 93, 165) if source == "llm" else (70, 70, 70)
    font = _font(round(10 * TEXT_SCALE), bold=True)
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, _ = box
    tw = int(draw.textlength(label, font=font))
    return _chip(canvas, (x1 - tw - 13, y0 + 5), [(label, color, font)], pad=4)


def build_fig2_mixed(out_name: str) -> None:
    """Build five 2x2 robot blocks: Human/Human over recognized LLM/LLM."""
    block_w = 2 * CELL
    header_h = 28
    gap = 10
    width = len(FIG2_ROBOTS) * block_w + (len(FIG2_ROBOTS) - 1) * gap
    height = header_h + 2 * CELL
    canvas = Image.new("RGB", (width, height), "white")
    robot_font = _font(round(15 * TEXT_SCALE), bold=True)
    cue_font = _font(round(11 * TEXT_SCALE), bold=True)
    cap_font = _font(FONT_SIZE)

    for robot_col, (platform, robot_label) in enumerate(FIG2_ROBOTS):
        bx = robot_col * (block_w + gap)
        _centered(ImageDraw.Draw(canvas), (bx, 0, bx + block_w, header_h), robot_label, robot_font)
        for pos, (idx, source) in enumerate(FIG2_MIXED_CUES[platform]):
            row, col = divmod(pos, 2)
            x0 = bx + col * CELL
            y0 = header_h + row * CELL
            canvas.paste(_cell(platform, idx, trajectory=True, source=source), (x0, y0))
            canvas = _draw_cue_name(
                canvas, (x0 + PAD, y0 + PAD), _cue_label(CUES[idx]), cue_font,
                platform=platform,
            )
            canvas = _source_badge(canvas, (x0, y0, x0 + CELL, y0 + CELL), source)
            key = (platform, idx)
            wrapped = _wrap(
                ImageDraw.Draw(canvas), CELL_CAPS[key], cap_font,
                CELL - 2 * PAD - 10, max_lines=3,
            )
            lines = [(line, CAP_RGB, cap_font) for line in wrapped]
            probe = ImageDraw.Draw(canvas)
            tw = max(int(probe.textlength(text, font=font)) for text, _, font in lines)
            chip_w = tw + 2 * PAD
            chip_h = (FONT_SIZE + 4) * len(lines) + 2 * PAD
            canvas = _chip(
                canvas,
                (x0 + (CELL - chip_w) // 2, y0 + CELL - chip_h - PAD),
                lines,
            )
        if robot_col < len(FIG2_ROBOTS) - 1:
            sx = bx + block_w + gap // 2
            ImageDraw.Draw(canvas).line((sx, 4, sx, height - 4), fill="#d0d0d0")

    out = FIGURES / out_name
    canvas.save(out, optimize=True)
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    build_paired("figure_rq2_all_morphologies_updated_tiago31.png", trajectory=True)
