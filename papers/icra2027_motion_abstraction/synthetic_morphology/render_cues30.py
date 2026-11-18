#!/usr/bin/env python3
"""30 paper cues × 3 Oddkin morphologies: Motion Code YAML + GIF gallery.

Few-shot style (raising_hand_greeting / wave_hi / bow_greeting / blow_kiss /
circle_temple_crazy) informs rematerialization patterns. Programs live in
yml/cues30_oddkin.yaml; this script executes structured plans and renders.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
XML_DIR = ROOT / "xml"
YML_OUT = ROOT / "yml" / "cues30_oddkin.yaml"
OUT_DIR = ROOT / "renders" / "cues30"
FPS = 11
W, H = 420, 320

CUES = [
    "nod_yes", "shake_head_no", "draw_circle_repeat", "gaze_shift_avoid", "fist_pump",
    "tilt_head_curious", "cover_yawn_sleepy", "rub_eye_tired", "point_you", "point_self",
    "beckon_come_here", "stop_palm_out", "salute", "raise_index_one_moment", "highfive_invite",
    "shrug_idk", "flex_bicep", "air_quotes", "arms_open_welcome", "hands_up_surrender",
    "stretch_arms_up", "raise_fists_victory", "cross_arms_defensive", "palms_together_thank_you",
    "facepalm", "depict_tree", "depict_big_bowl", "depict_lion", "depict_small_object",
    "depict_flying_bird",
]

MORPHS = [
    ("gcr_alpha0", "α=0", (0.0, 0.0, 0.14), 1.45, 0.12),
    ("gcr_alpha05", "α=0.5", (0.0, 0.0, 0.30), 1.75, 0.11),
    ("gcr_alpha1", "α=1.0", (0.0, 0.0, 0.40), 2.05, 0.13),
]


# ---------------------------------------------------------------------------
# homes / pose libraries
# ---------------------------------------------------------------------------

def home0(extra=None):
    j = {"hip0": 0.2, "knee0": -0.6, "hip1": 0.15, "knee1": -0.55,
         "hip2": -0.1, "knee2": -0.65, "crown_spin": 0.0}
    if extra:
        j.update(extra)
    return j


def home05(extra=None):
    j = {
        "flap_l": 0.2, "flap_r": -0.2,
        "v1_pitch": 0.1, "v1_yaw": 0.0, "v2_pitch": 0.08, "v2_twist": 0.15,
        "collar_spin": 0.0, "v3_pitch": 0.05,
        "pin_r0": 0.35, "pin_r1": -0.7, "pin_r2": 0.1,
        "pin_l0": -0.4, "pin_l1": -0.3, "antenna": 0.3,
    }
    if extra:
        j.update(extra)
    return j


def home1(extra=None):
    j = {
        "trunk1_yaw": 0.0, "trunk1_pitch": 0.0,
        "trunk2_yaw": 0.0, "trunk2_pitch": 0.0,
        "sat_spin": 0.0, "trunk3_pitch": 0.0, "trunk3_roll": 0.0,
        "bl0": 0.25, "bl1": 0.25, "bl2": 0.0,
        "br0": -0.25, "br1": 0.30, "br2": 0.15, "br3": 0.0,
        "claw0": 0.25, "claw1": -0.25, "claw2": -0.05,
        "neck_yaw": 0.0, "neck_pitch": 0.0, "head_roll": 0.0,
    }
    if extra:
        j.update(extra)
    return j


# α0.5 spine heights
SP05 = {
    "mid": {"v1_pitch": 0.1, "v2_pitch": 0.08, "v3_pitch": 0.05},
    "high": {"v1_pitch": -0.45, "v2_pitch": -0.4, "v3_pitch": -0.5},
    "low": {"v1_pitch": 0.75, "v2_pitch": 0.65, "v3_pitch": 0.7},
}

# α1 arm presets
A1 = {
    "still": {},
    "paddle_up": {"bl0": 0.15, "bl1": -1.5, "bl2": 1.1},
    "paddle_front": {"bl0": 0.1, "bl1": -0.15, "bl2": 1.2},
    "paddle_out": {"bl0": 0.7, "bl1": -0.4, "bl2": 0.8},
    "paddle_in": {"bl0": -0.5, "bl1": -0.6, "bl2": 0.5},
    "claw_front": {"br0": -0.1, "br1": 0.1, "br2": 0.0},
    "claw_up": {"br0": 0.1, "br1": -1.5, "br2": -0.9},
    "claw_out": {"br0": -0.8, "br1": -0.5, "br2": -0.3},
    "claw_in": {"br0": 0.7, "br1": -1.1, "br2": -0.5},
    "claw_back": {"br0": 0.4, "br1": -0.8, "br2": -0.4},
    "both_up": {"bl0": 0.15, "bl1": -1.5, "bl2": 1.0, "br0": 0.1, "br1": -1.5, "br2": -0.9},
    "both_out": {"bl0": 0.7, "bl1": -0.35, "bl2": 0.9, "br0": -0.7, "br1": -0.35, "br2": -0.2},
    "grip_close": {"claw0": 1.0, "claw1": -1.0, "claw2": -0.6},
    "grip_open": {"claw0": 0.15, "claw1": -0.15, "claw2": 0.0},
    "grip_pinch": {"claw0": 0.85, "claw1": -0.85, "claw2": -0.4},
}


def _merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# ---------------------------------------------------------------------------
# Motion Code strings (documentation / YAML) + executable plans
# ---------------------------------------------------------------------------

def build_catalog():
    """Return {cue: {morph: {code, plan}}} where plan is list of ops."""
    cat = {}

    def put(cue, morph, code, plan):
        cat.setdefault(cue, {})[morph] = {"code": " ".join(code.split()), "plan": plan}

    # ---- nod_yes ----
    put("nod_yes", "gcr_alpha0",
        "path(base, arc, r=0, deg=±12, rep=3, spd=1.5)",
        [{"op": "osc", "base_yaw": 0.21, "rep": 3, "n": 30, "home": "0"}])
    put("nod_yes", "gcr_alpha05",
        "pose(torso=mid) + move(torso, pitch, ±12, rep=3, spd=1.5)",
        [{"op": "pose", "home": "05", "joints": SP05["mid"], "n": 6},
         {"op": "osc", "joints": {"v1_pitch": 0.35, "v2_pitch": 0.30, "v3_pitch": 0.35},
          "center": SP05["mid"], "rep": 3, "n": 30, "home": "05"}])
    put("nod_yes", "gcr_alpha1",
        "pose(head=center, torso=mid) + move(head, tilt, [-12,12], rep=3, spd=1.5)",
        [{"op": "pose", "home": "1", "n": 6},
         {"op": "osc", "joints": {"neck_pitch": 0.35}, "rep": 3, "n": 30, "home": "1"}])

    # ---- shake_head_no ----
    put("shake_head_no", "gcr_alpha0",
        "move(base, yaw, ±25, rep=3, spd=2.0)",
        [{"op": "osc", "base_yaw": 0.44, "rep": 3, "n": 30, "home": "0"}])
    put("shake_head_no", "gcr_alpha05",
        "path(torso, arc, r=0, deg=±25, rep=3)",
        [{"op": "osc", "joints": {"v1_yaw": 0.45, "v2_twist": 0.35}, "rep": 3, "n": 30, "home": "05"}])
    put("shake_head_no", "gcr_alpha1",
        "move(head, pan, ±20, rep=3, spd=2.0)",
        [{"op": "osc", "joints": {"neck_yaw": 0.45}, "rep": 3, "n": 30, "home": "1"}])

    # ---- draw_circle_repeat ----
    put("draw_circle_repeat", "gcr_alpha0",
        "path(base, arc, r=0.15, deg=360, rep=2, spd=1.4)",
        [{"op": "base_arc", "r": 0.15, "turns": 2, "n": 40, "home": "0"}])
    put("draw_circle_repeat", "gcr_alpha05",
        "pose(long=front) + path(ee_long, arc, plane=xy, r=0.08, sweep=360, rep=2)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.1, "pin_r1": -0.4}, "n": 6},
         {"op": "osc", "joints": {"pin_r0": 0.5, "pin_r2": 0.6}, "rep": 2, "n": 36, "home": "05",
          "base_joints": {"pin_r0": 0.1, "pin_r1": -0.4}}])
    put("draw_circle_repeat", "gcr_alpha1",
        "pose(claw=front) + path(ee_claw, arc, plane=xy, r=0.08, sweep=360, rep=2)",
        [{"op": "pose", "home": "1", "joints": A1["claw_front"], "n": 6},
         {"op": "osc", "joints": {"br0": 0.45, "br3": 0.5}, "rep": 2, "n": 36, "home": "1",
          "base_joints": A1["claw_front"]}])

    # ---- gaze_shift_avoid ----
    put("gaze_shift_avoid", "gcr_alpha0",
        "pose(dir=left) + pose(dir=right) + pose(dir=front)",
        [{"op": "pose", "home": "0", "base": (0, 0, 0.9), "n": 10},
         {"op": "pose", "home": "0", "base": (0, 0, -0.9), "n": 10},
         {"op": "pose", "home": "0", "base": (0, 0, 0), "n": 8}])
    put("gaze_shift_avoid", "gcr_alpha05",
        "pose(dir=left) + pose(dir=right) + pose(dir=front)",
        [{"op": "pose", "home": "05", "joints": {"v1_yaw": 0.55, "v2_twist": 0.5}, "n": 10},
         {"op": "pose", "home": "05", "joints": {"v1_yaw": -0.55, "v2_twist": -0.5}, "n": 10},
         {"op": "pose", "home": "05", "n": 8}])
    put("gaze_shift_avoid", "gcr_alpha1",
        "pose(head=left) + pose(head=right) + pose(head=center)",
        [{"op": "pose", "home": "1", "joints": {"neck_yaw": 0.7}, "n": 10},
         {"op": "pose", "home": "1", "joints": {"neck_yaw": -0.7}, "n": 10},
         {"op": "pose", "home": "1", "n": 8}])

    # ---- fist_pump ----
    put("fist_pump", "gcr_alpha0",
        "pose(leg0=raise) + move(hip0, yaw, [0,40], rep=3, spd=2.2)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.8, "knee0": -0.2}, "n": 6},
         {"op": "osc", "joints": {"hip0": 0.45, "knee0": 0.25}, "rep": 3, "n": 28, "home": "0",
          "base_joints": {"hip0": 0.8, "knee0": -0.2}}])
    put("fist_pump", "gcr_alpha05",
        "pose(long=up) + move(long_pincer, pitch, [0,30], rep=3, spd=2.2)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.15, "pin_r1": -1.55}, "n": 6},
         {"op": "osc", "joints": {"pin_r1": 0.35}, "rep": 3, "n": 28, "home": "05",
          "base_joints": {"pin_r0": 0.15, "pin_r1": -1.55}}])
    put("fist_pump", "gcr_alpha1",
        "pose(claw=up, grip=close) + move(claw, pitch, [0,30], rep=3, spd=2.2)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_up"], A1["grip_close"]), "n": 6},
         {"op": "osc", "joints": {"br1": 0.35}, "rep": 3, "n": 28, "home": "1",
          "base_joints": _merge(A1["claw_up"], A1["grip_close"])}])

    # ---- tilt_head_curious ----
    put("tilt_head_curious", "gcr_alpha0",
        "pose(dir=left) + hold(t=1.0)",
        [{"op": "pose", "home": "0", "base": (0, 0, 0.7), "joints": {"crown_spin": 0.5}, "n": 22}])
    put("tilt_head_curious", "gcr_alpha05",
        "pose(dir=left, antenna=forward) + hold(t=1.0)",
        [{"op": "pose", "home": "05", "joints": {"v1_yaw": 0.5, "v2_twist": 0.4, "antenna": -0.4}, "n": 22}])
    put("tilt_head_curious", "gcr_alpha1",
        "pose(head=left) + move(head, roll, +20, hold=1.0)",
        [{"op": "pose", "home": "1", "joints": {"neck_yaw": 0.45, "head_roll": 0.5}, "n": 22}])

    # ---- cover_yawn_sleepy ----
    put("cover_yawn_sleepy", "gcr_alpha0",
        "path(base, arc, r=0.25, deg=15, spd=0.8) + path(base, arc, r=0.25, deg=-30, spd=0.5, hold=1.2)",
        [{"op": "base_arc", "r": 0.22, "turns": 0.08, "n": 12, "home": "0"},
         {"op": "base_arc", "r": 0.22, "turns": -0.12, "n": 16, "home": "0", "hold_end": 12}])
    put("cover_yawn_sleepy", "gcr_alpha05",
        "pose(torso=low, long=in) + hold(t=1.2)",
        [{"op": "lerp", "home": "05", "to": _merge(SP05["low"], {"pin_r0": -0.4, "pin_r1": -1.0}), "n": 16},
         {"op": "pose", "home": "05", "joints": _merge(SP05["low"], {"pin_r0": -0.4, "pin_r1": -1.0}), "n": 14}])
    put("cover_yawn_sleepy", "gcr_alpha1",
        "pose(claw=in, paddle=still, head=down, torso=low) + hold(t=1.2)",
        [{"op": "lerp", "home": "1",
          "to": _merge(A1["claw_in"], {"trunk1_pitch": 0.4, "trunk2_pitch": 0.35, "neck_pitch": 0.5}), "n": 16},
         {"op": "pose", "home": "1",
          "joints": _merge(A1["claw_in"], {"trunk1_pitch": 0.4, "trunk2_pitch": 0.35, "neck_pitch": 0.5}), "n": 14}])

    # ---- rub_eye_tired ----
    put("rub_eye_tired", "gcr_alpha0",
        "pose(leg0=present) + move(hip0, yaw, ±15, rep=4, spd=1.2)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.7, "knee0": -0.25}, "n": 6},
         {"op": "osc", "joints": {"hip0": 0.25}, "rep": 4, "n": 32, "home": "0",
          "base_joints": {"hip0": 0.7, "knee0": -0.25}}])
    put("rub_eye_tired", "gcr_alpha05",
        "pose(long=in) + move(long_pincer, tip, ±20, rep=4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": -0.3, "pin_r1": -1.1}, "n": 6},
         {"op": "osc", "joints": {"pin_r2": 0.5}, "rep": 4, "n": 32, "home": "05",
          "base_joints": {"pin_r0": -0.3, "pin_r1": -1.1}}])
    put("rub_eye_tired", "gcr_alpha1",
        "pose(claw=in, head=center) + move(wrist_c, roll, ±25, rep=4)",
        [{"op": "pose", "home": "1", "joints": A1["claw_in"], "n": 6},
         {"op": "osc", "joints": {"br3": 0.55}, "rep": 4, "n": 32, "home": "1",
          "base_joints": A1["claw_in"]}])

    # ---- point_you ----
    put("point_you", "gcr_alpha0",
        "pose(dir=front, leg0=present) + hold(t=1.4)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.95, "knee0": -0.1}, "n": 24}])
    put("point_you", "gcr_alpha05",
        "pose(dir=front, long=front, long_grip=edge) + hold(t=1.4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.05, "pin_r1": -0.25, "pin_r2": 0.0}, "n": 24}])
    put("point_you", "gcr_alpha1",
        "pose(dir=front, claw=front, grip=pinch, paddle=still) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_front"], A1["grip_pinch"]), "n": 24}])

    # ---- point_self ----
    put("point_self", "gcr_alpha0",
        "pose(dir=back, leg0=present) + hold(t=1.4)",
        [{"op": "pose", "home": "0", "base": (0, 0, math.pi), "joints": {"hip0": 0.7}, "n": 24}])
    put("point_self", "gcr_alpha05",
        "pose(long=back) + hold(t=1.4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 1.0, "pin_r1": -0.8}, "n": 24}])
    put("point_self", "gcr_alpha1",
        "pose(claw=back, grip=pinch) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_back"], A1["grip_pinch"]), "n": 24}])

    # ---- beckon_come_here ----
    put("beckon_come_here", "gcr_alpha0",
        "pose(leg0=raise) + move(knee0, pitch, ±25, rep=3) + path(base, line, x=+0.08)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.7, "knee0": -0.3}, "n": 4},
         {"op": "osc", "joints": {"knee0": 0.4}, "rep": 3, "n": 24, "home": "0",
          "base_joints": {"hip0": 0.7, "knee0": -0.3}},
         {"op": "base_line", "x": 0.08, "n": 10, "home": "0", "joints": {"hip0": 0.7, "knee0": -0.3}}])
    put("beckon_come_here", "gcr_alpha05",
        "pose(long=front) + move(long_pincer, tip, ±30, rep=3)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.2, "pin_r1": -0.5}, "n": 4},
         {"op": "osc", "joints": {"pin_r2": 0.7}, "rep": 3, "n": 28, "home": "05",
          "base_joints": {"pin_r0": 0.2, "pin_r1": -0.5}}])
    put("beckon_come_here", "gcr_alpha1",
        "pose(claw=front, grip=open, elbow_c=T) + move(claw, pitch, ±25, rep=3)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_front"], A1["grip_open"], {"br1": -0.7}), "n": 4},
         {"op": "osc", "joints": {"br1": 0.4}, "rep": 3, "n": 28, "home": "1",
          "base_joints": _merge(A1["claw_front"], A1["grip_open"], {"br1": -0.7})}])

    # ---- stop_palm_out ----
    put("stop_palm_out", "gcr_alpha0",
        "path(base, line, x=+0.12) + hold(t=1.5)",
        [{"op": "base_line", "x": 0.12, "n": 12, "home": "0"},
         {"op": "pose", "home": "0", "base": (0.12, 0, 0), "n": 16}])
    put("stop_palm_out", "gcr_alpha05",
        "pose(torso=high, long=front, long_grip=flat) + hold(t=1.5)",
        [{"op": "pose", "home": "05",
          "joints": _merge(SP05["high"], {"pin_r0": 0.05, "pin_r1": -0.2, "pin_r2": 0.0}), "n": 24}])
    put("stop_palm_out", "gcr_alpha1",
        "pose(paddle=front, palm=palmfront, claw=still, torso=mid) + hold(t=1.5)",
        [{"op": "pose", "home": "1", "joints": A1["paddle_front"], "n": 24}])

    # ---- salute ----
    put("salute", "gcr_alpha0",
        "pose(leg0=present) + hold(t=1.4)",
        [{"op": "lerp", "home": "0", "to": {"hip0": 1.0, "knee0": -0.05}, "n": 12},
         {"op": "pose", "home": "0", "joints": {"hip0": 1.0, "knee0": -0.05}, "n": 14}])
    put("salute", "gcr_alpha05",
        "pose(long=up, short=still) + hold(t=1.4)",
        [{"op": "lerp", "home": "05", "to": {"pin_r0": 0.1, "pin_r1": -1.7}, "n": 12},
         {"op": "pose", "home": "05", "joints": {"pin_r0": 0.1, "pin_r1": -1.7}, "n": 14}])
    put("salute", "gcr_alpha1",
        "pose(claw=up, grip=edge-like, head=center) + hold(t=1.4)",
        [{"op": "lerp", "home": "1", "to": _merge(A1["claw_up"], A1["grip_pinch"], {"br0": 0.5}), "n": 12},
         {"op": "pose", "home": "1", "joints": _merge(A1["claw_up"], A1["grip_pinch"], {"br0": 0.5}), "n": 14}])

    # ---- raise_index_one_moment ----
    put("raise_index_one_moment", "gcr_alpha0",
        "pose(leg0=raise) + hold(t=1.3)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.85, "knee0": -0.35}, "n": 22}])
    put("raise_index_one_moment", "gcr_alpha05",
        "pose(short=up) + hold(t=1.3)",
        [{"op": "pose", "home": "05", "joints": {"pin_l0": -0.2, "pin_l1": -1.3}, "n": 22}])
    put("raise_index_one_moment", "gcr_alpha1",
        "pose(claw=up, grip=pinch, elbow_c=T) + hold(t=1.3)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_up"], A1["grip_pinch"]), "n": 22}])

    # ---- highfive_invite (few-shot style) ----
    put("highfive_invite", "gcr_alpha0",
        "path(base, arc, r=0, deg=360, spd=2.5) + path(base, line, x=+0.20, hold=1.0) + pose(leg0=present)",
        [{"op": "base_spin", "turns": 1, "n": 24, "home": "0"},
         {"op": "base_line", "x": 0.20, "n": 14, "home": "0", "joints": {"hip0": 0.95, "knee0": -0.1}},
         {"op": "pose", "home": "0", "base": (0.20, 0, 0), "joints": {"hip0": 0.95, "knee0": -0.1}, "n": 10}])
    put("highfive_invite", "gcr_alpha05",
        "pose(torso=high, long=front) + path(base, line, x=+0.15, hold=1.2)",
        [{"op": "lerp", "home": "05",
          "to": _merge(SP05["high"], {"pin_r0": 0.05, "pin_r1": -0.25}), "n": 12},
         {"op": "base_line", "x": 0.15, "n": 12, "home": "05",
          "joints": _merge(SP05["high"], {"pin_r0": 0.05, "pin_r1": -0.25})},
         {"op": "pose", "home": "05", "base": (0.15, 0, 0),
          "joints": _merge(SP05["high"], {"pin_r0": 0.05, "pin_r1": -0.25}), "n": 10}])
    put("highfive_invite", "gcr_alpha1",
        "pose(claw=front, grip=open) + path(ee_claw, line, axis=x, dist=+0.15, hold=1.0)",
        [{"op": "lerp", "home": "1", "to": _merge(A1["claw_front"], A1["grip_open"]), "n": 10},
         {"op": "base_line", "x": 0.12, "n": 12, "home": "1",
          "joints": _merge(A1["claw_front"], A1["grip_open"])},
         {"op": "pose", "home": "1", "base": (0.12, 0, 0),
          "joints": _merge(A1["claw_front"], A1["grip_open"]), "n": 10}])

    # ---- shrug_idk ----
    put("shrug_idk", "gcr_alpha0",
        "pose(leg0=raise, leg1=raise, leg2=raise) + move(base, yaw, ±10, rep=2)",
        [{"op": "pose", "home": "0",
          "joints": {"hip0": 0.7, "hip1": 0.6, "hip2": 0.5, "knee0": -0.3, "knee1": -0.3, "knee2": -0.3}, "n": 8},
         {"op": "osc", "base_yaw": 0.18, "rep": 2, "n": 24, "home": "0",
          "joints": {"hip0": 0.7, "hip1": 0.6, "hip2": 0.5}}])
    put("shrug_idk", "gcr_alpha05",
        "pose(long=out, short=out) + move(torso, height, high, hold=0.4) + move(long_pincer, yaw, ±20) + move(short_pincer, yaw, ±20)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 1.0, "pin_l0": -1.0}, "n": 6},
         {"op": "lerp", "home": "05", "from_joints": {"pin_r0": 1.0, "pin_l0": -1.0},
          "to": _merge(SP05["high"], {"pin_r0": 1.0, "pin_l0": -1.0}), "n": 10},
         {"op": "osc", "joints": {"pin_r0": 0.35, "pin_l0": 0.35}, "rep": 2, "n": 24, "home": "05",
          "base_joints": _merge(SP05["high"], {"pin_r0": 1.0, "pin_l0": -1.0})}])
    put("shrug_idk", "gcr_alpha1",
        "pose(paddle=out, claw=out, palm=palmup) + move(torso, pitch, +15, hold=0.4) + move(paddle, yaw, ±15) + move(claw, yaw, ±15)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["both_out"], {"bl2": 0.3}), "n": 6},
         {"op": "pose", "home": "1", "joints": _merge(A1["both_out"], {"trunk1_pitch": 0.25, "bl2": 0.3}), "n": 8},
         {"op": "osc", "joints": {"bl0": 0.3, "br0": 0.3}, "rep": 2, "n": 24, "home": "1",
          "base_joints": _merge(A1["both_out"], {"trunk1_pitch": 0.25, "bl2": 0.3})}])

    # ---- flex_bicep ----
    put("flex_bicep", "gcr_alpha0",
        "pose(leg0=present) + hold(t=1.3)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.9, "knee0": -1.2}, "n": 22}])
    put("flex_bicep", "gcr_alpha05",
        "pose(long=up) + move(long_pincer, pitch, -40, hold=1.2)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.3, "pin_r1": -1.8, "pin_r2": 0.6}, "n": 22}])
    put("flex_bicep", "gcr_alpha1",
        "pose(claw=out, elbow_c=T, grip=close) + hold(t=1.3)",
        [{"op": "pose", "home": "1",
          "joints": _merge(A1["claw_out"], A1["grip_close"], {"br1": -1.6, "br2": -1.2}), "n": 22}])

    # ---- air_quotes ----
    put("air_quotes", "gcr_alpha0",
        "pose(leg0=raise, leg1=raise) + move(hip0, yaw, ±20, rep=2) + move(hip1, yaw, ±20, rep=2)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.7, "hip1": 0.6, "knee0": -0.3, "knee1": -0.3}, "n": 6},
         {"op": "osc", "joints": {"hip0": 0.35, "hip1": 0.35}, "rep": 2, "n": 28, "home": "0",
          "base_joints": {"hip0": 0.7, "hip1": 0.6, "knee0": -0.3, "knee1": -0.3}}])
    put("air_quotes", "gcr_alpha05",
        "pose(long=up, short=up) + move(long_pincer, tip, ±25, rep=2) + move(short_pincer, pitch, ±25, rep=2)",
        [{"op": "pose", "home": "05",
          "joints": {"pin_r0": 0.2, "pin_r1": -1.4, "pin_l0": -0.2, "pin_l1": -1.2}, "n": 6},
         {"op": "osc", "joints": {"pin_r2": 0.55, "pin_l1": 0.4}, "rep": 2, "n": 28, "home": "05",
          "base_joints": {"pin_r0": 0.2, "pin_r1": -1.4, "pin_l0": -0.2, "pin_l1": -1.2}}])
    put("air_quotes", "gcr_alpha1",
        "pose(claw=up, paddle=up, grip=pinch) + move(wrist_c, roll, ±30, rep=2) + move(wrist_p, roll, ±30, rep=2)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["both_up"], A1["grip_pinch"]), "n": 6},
         {"op": "osc", "joints": {"br3": 0.6, "bl2": 0.6}, "rep": 2, "n": 28, "home": "1",
          "base_joints": _merge(A1["both_up"], A1["grip_pinch"])}])

    # ---- arms_open_welcome ----
    put("arms_open_welcome", "gcr_alpha0",
        "pose(leg0=raise, leg1=raise, leg2=raise) + hold(t=1.4)",
        [{"op": "pose", "home": "0",
          "joints": {"hip0": 0.8, "hip1": 0.7, "hip2": 0.7, "knee0": -0.25, "knee1": -0.25, "knee2": -0.25}, "n": 24}])
    put("arms_open_welcome", "gcr_alpha05",
        "pose(long=out, short=out, torso=mid) + hold(t=1.4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 1.1, "pin_l0": -1.1, "pin_r1": -0.5}, "n": 24}])
    put("arms_open_welcome", "gcr_alpha1",
        "pose(paddle=out, claw=out, palm=palmup) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["both_out"], {"bl2": 0.2}), "n": 24}])

    # ---- hands_up_surrender ----
    put("hands_up_surrender", "gcr_alpha0",
        "pose(leg0=raise, leg1=raise, leg2=raise) + path(base, line, x=-0.15) + hold(t=1.3)",
        [{"op": "pose", "home": "0",
          "joints": {"hip0": 0.9, "hip1": 0.85, "hip2": 0.85, "knee0": -0.15, "knee1": -0.15, "knee2": -0.15}, "n": 8},
         {"op": "base_line", "x": -0.15, "n": 12, "home": "0",
          "joints": {"hip0": 0.9, "hip1": 0.85, "hip2": 0.85}},
         {"op": "pose", "home": "0", "base": (-0.15, 0, 0),
          "joints": {"hip0": 0.9, "hip1": 0.85, "hip2": 0.85}, "n": 10}])
    put("hands_up_surrender", "gcr_alpha05",
        "pose(torso=high, long=up, short=up) + path(base, line, x=-0.15) + hold(t=1.3)",
        [{"op": "lerp", "home": "05",
          "to": _merge(SP05["high"], {"pin_r0": 0.15, "pin_r1": -1.6, "pin_l0": -0.2, "pin_l1": -1.3}), "n": 12},
         {"op": "base_line", "x": -0.15, "n": 12, "home": "05",
          "joints": _merge(SP05["high"], {"pin_r0": 0.15, "pin_r1": -1.6, "pin_l0": -0.2, "pin_l1": -1.3})},
         {"op": "pose", "home": "05", "base": (-0.15, 0, 0),
          "joints": _merge(SP05["high"], {"pin_r0": 0.15, "pin_r1": -1.6, "pin_l0": -0.2, "pin_l1": -1.3}), "n": 8}])
    put("hands_up_surrender", "gcr_alpha1",
        "pose(paddle=up, claw=up, grip=open) + path(base, line, x=-0.12) + hold(t=1.3)",
        [{"op": "lerp", "home": "1", "to": _merge(A1["both_up"], A1["grip_open"]), "n": 12},
         {"op": "base_line", "x": -0.12, "n": 12, "home": "1",
          "joints": _merge(A1["both_up"], A1["grip_open"])},
         {"op": "pose", "home": "1", "base": (-0.12, 0, 0),
          "joints": _merge(A1["both_up"], A1["grip_open"]), "n": 8}])

    # ---- stretch_arms_up ----
    put("stretch_arms_up", "gcr_alpha0",
        "pose(leg0=raise, leg1=raise, leg2=raise, crown=half) + hold(t=1.4)",
        [{"op": "pose", "home": "0",
          "joints": {"hip0": 0.95, "hip1": 0.9, "hip2": 0.9, "knee0": -0.1, "knee1": -0.1, "knee2": -0.1,
                     "crown_spin": math.pi}, "n": 24}])
    put("stretch_arms_up", "gcr_alpha05",
        "pose(torso=high, long=up, short=up) + hold(t=1.4)",
        [{"op": "pose", "home": "05",
          "joints": _merge(SP05["high"], {"pin_r0": 0.1, "pin_r1": -1.7, "pin_l0": -0.15, "pin_l1": -1.4}), "n": 24}])
    put("stretch_arms_up", "gcr_alpha1",
        "pose(paddle=up, claw=up, torso=high) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": A1["both_up"], "n": 24}])

    # ---- raise_fists_victory ----
    put("raise_fists_victory", "gcr_alpha0",
        "pose(leg0=present, leg1=raise) + hold(t=1.4)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 1.0, "knee0": -0.05, "hip1": 0.85, "knee1": -0.15}, "n": 24}])
    put("raise_fists_victory", "gcr_alpha05",
        "pose(long=up, short=up, long_grip=close) + hold(t=1.4)",
        [{"op": "pose", "home": "05",
          "joints": {"pin_r0": 0.15, "pin_r1": -1.65, "pin_r2": 0.9, "pin_l0": -0.2, "pin_l1": -1.35}, "n": 24}])
    put("raise_fists_victory", "gcr_alpha1",
        "pose(paddle=up, claw=up, grip=close) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["both_up"], A1["grip_close"]), "n": 24}])

    # ---- cross_arms_defensive ----
    put("cross_arms_defensive", "gcr_alpha0",
        "pose(leg0=plant, leg1=plant, leg2=plant) + hold(t=1.4)",
        [{"op": "pose", "home": "0",
          "joints": {"hip0": -0.2, "hip1": -0.2, "hip2": -0.2, "knee0": -1.2, "knee1": -1.2, "knee2": -1.2}, "n": 24}])
    put("cross_arms_defensive", "gcr_alpha05",
        "pose(long=in, short=in, torso=mid) + hold(t=1.4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": -0.8, "pin_l0": 0.8, "pin_r1": -0.9, "pin_l1": -0.9}, "n": 24}])
    put("cross_arms_defensive", "gcr_alpha1",
        "pose(paddle=in, claw=in) + hold(t=1.4)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["paddle_in"], A1["claw_in"]), "n": 24}])

    # ---- palms_together_thank_you ----
    put("palms_together_thank_you", "gcr_alpha0",
        "pose(dir=front) + path(base, line, x=+0.08) + move(base, yaw, 0, hold=1.2)",
        [{"op": "base_line", "x": 0.08, "n": 10, "home": "0"},
         {"op": "pose", "home": "0", "base": (0.08, 0, 0), "joints": {"crown_spin": 0.3}, "n": 16}])
    put("palms_together_thank_you", "gcr_alpha05",
        "pose(long=front, short=front) + move(torso, pitch, +20, hold=1.0)",
        [{"op": "pose", "home": "05",
          "joints": {"pin_r0": 0.2, "pin_l0": -0.2, "pin_r1": -0.4, "pin_l1": -0.5,
                     "v1_pitch": 0.45, "v2_pitch": 0.4}, "n": 24}])
    put("palms_together_thank_you", "gcr_alpha1",
        "pose(paddle=front, claw=front, palm=palmin) + move(torso, pitch, +20, hold=1.0)",
        [{"op": "pose", "home": "1",
          "joints": _merge(A1["paddle_front"], A1["claw_front"], {"trunk1_pitch": 0.35, "bl2": 0.0}), "n": 24}])

    # ---- facepalm ----
    put("facepalm", "gcr_alpha0",
        "pose(leg0=present) + path(base, line, x=-0.05) + hold(t=1.3)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.6, "knee0": -0.4}, "n": 8},
         {"op": "base_line", "x": -0.05, "n": 8, "home": "0", "joints": {"hip0": 0.6, "knee0": -0.4}},
         {"op": "pose", "home": "0", "base": (-0.05, 0, 0), "joints": {"hip0": 0.6, "knee0": -0.4}, "n": 12}])
    put("facepalm", "gcr_alpha05",
        "pose(long=in, torso=low) + hold(t=1.3)",
        [{"op": "pose", "home": "05",
          "joints": _merge(SP05["low"], {"pin_r0": -0.5, "pin_r1": -1.2}), "n": 24}])
    put("facepalm", "gcr_alpha1",
        "pose(claw=in, head=down, paddle=still) + hold(t=1.3)",
        [{"op": "pose", "home": "1",
          "joints": _merge(A1["claw_in"], {"neck_pitch": 0.55, "trunk1_pitch": 0.25}), "n": 24}])

    # ---- depict_tree ----
    put("depict_tree", "gcr_alpha0",
        "path(base, line, x=+0.12) + pose(leg0=present) + pose(leg1=raise) + pose(leg2=raise) + hold(t=1.0)",
        [{"op": "base_line", "x": 0.12, "n": 10, "home": "0"},
         {"op": "pose", "home": "0", "base": (0.12, 0, 0),
          "joints": {"hip0": 0.95, "knee0": -0.1, "hip1": 0.7, "hip2": 0.7}, "n": 18}])
    put("depict_tree", "gcr_alpha05",
        "pose(torso=high) + pose(long=up, short=out) + hold(t=1.2)",
        [{"op": "pose", "home": "05",
          "joints": _merge(SP05["high"], {"pin_r0": 0.1, "pin_r1": -1.6, "pin_l0": -1.0}), "n": 24}])
    put("depict_tree", "gcr_alpha1",
        "pose(torso=high, paddle=up, claw=out) + hold(t=1.2)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["paddle_up"], A1["claw_out"]), "n": 24}])

    # ---- depict_big_bowl ----
    put("depict_big_bowl", "gcr_alpha0",
        "path(base, arc, r=0.28, deg=-180, spd=0.9) + hold(t=0.8)",
        [{"op": "base_arc", "r": 0.28, "turns": -0.5, "n": 28, "home": "0", "hold_end": 8}])
    put("depict_big_bowl", "gcr_alpha05",
        "pose(long=out, short=out) + path(ee_long, arc, plane=yz, r=0.12, sweep=-120)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.9, "pin_l0": -0.9}, "n": 6},
         {"op": "osc", "joints": {"pin_r1": 0.6, "v1_pitch": 0.3}, "rep": 1, "n": 28, "home": "05",
          "base_joints": {"pin_r0": 0.9, "pin_l0": -0.9}, "phase": "half"}])
    put("depict_big_bowl", "gcr_alpha1",
        "pose(paddle=out, claw=out) + path(ee_claw, arc, plane=yz, r=0.14, sweep=-120)",
        [{"op": "pose", "home": "1", "joints": A1["both_out"], "n": 6},
         {"op": "osc", "joints": {"br1": 0.55, "bl1": 0.55}, "rep": 1, "n": 28, "home": "1",
          "base_joints": A1["both_out"], "phase": "half"}])

    # ---- depict_lion (few-shot / contextualized) ----
    put("depict_lion", "gcr_alpha0",
        "path(base, line, x=-0.10) + path(base, line, x=+0.05) + path(base, line, x=+0.18) + pose(leg0=present)",
        [{"op": "base_line", "x": -0.10, "n": 10, "home": "0"},
         {"op": "base_line", "x": 0.05, "n": 8, "home": "0", "start_x": -0.10},
         {"op": "base_line", "x": 0.18, "n": 12, "home": "0", "start_x": -0.05,
          "joints": {"hip0": 0.95, "knee0": -0.1}},
         {"op": "pose", "home": "0", "base": (0.13, 0, 0), "joints": {"hip0": 0.95, "knee0": -0.1}, "n": 8}])
    put("depict_lion", "gcr_alpha05",
        "pose(long=front) + path(base, line, x=-0.08) + path(ee_long, line, axis=x, dist=+0.15)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.1, "pin_r1": -0.5}, "n": 6},
         {"op": "base_line", "x": -0.08, "n": 8, "home": "05", "joints": {"pin_r0": 0.1, "pin_r1": -0.5}},
         {"op": "lerp", "home": "05", "base": (-0.08, 0, 0),
          "from_joints": {"pin_r0": 0.1, "pin_r1": -0.5},
          "to": {"pin_r0": 0.05, "pin_r1": -0.15}, "n": 14},
         {"op": "pose", "home": "05", "base": (-0.08, 0, 0), "joints": {"pin_r0": 0.05, "pin_r1": -0.15}, "n": 8}])
    put("depict_lion", "gcr_alpha1",
        "pose(claw=front, grip=open, torso=low) + path(base, line, x=-0.10) + path(ee_claw, line, axis=x, dist=+0.20)",
        [{"op": "pose", "home": "1",
          "joints": _merge(A1["claw_front"], A1["grip_open"], {"trunk1_pitch": 0.35}), "n": 6},
         {"op": "base_line", "x": -0.10, "n": 8, "home": "1",
          "joints": _merge(A1["claw_front"], A1["grip_open"], {"trunk1_pitch": 0.35})},
         {"op": "base_line", "x": 0.22, "n": 14, "home": "1", "start_x": -0.10,
          "joints": _merge(A1["claw_front"], A1["grip_open"], {"br1": 0.05, "trunk1_pitch": 0.2})},
         {"op": "pose", "home": "1", "base": (0.12, 0, 0),
          "joints": _merge(A1["claw_front"], A1["grip_open"], {"br1": 0.05}), "n": 8}])

    # ---- depict_small_object ----
    put("depict_small_object", "gcr_alpha0",
        "pose(leg0=raise) + move(knee0, pitch, ±12, rep=2) + hold(t=0.8)",
        [{"op": "pose", "home": "0", "joints": {"hip0": 0.55, "knee0": -0.5}, "n": 6},
         {"op": "osc", "joints": {"knee0": 0.25}, "rep": 2, "n": 20, "home": "0",
          "base_joints": {"hip0": 0.55, "knee0": -0.5}},
         {"op": "pose", "home": "0", "joints": {"hip0": 0.55, "knee0": -0.5}, "n": 8}])
    put("depict_small_object", "gcr_alpha05",
        "pose(short=front) + move(short_pincer, pitch, ±15, rep=2) + hold(t=0.8)",
        [{"op": "pose", "home": "05", "joints": {"pin_l0": -0.3, "pin_l1": -0.5}, "n": 6},
         {"op": "osc", "joints": {"pin_l1": 0.3}, "rep": 2, "n": 20, "home": "05",
          "base_joints": {"pin_l0": -0.3, "pin_l1": -0.5}},
         {"op": "pose", "home": "05", "joints": {"pin_l0": -0.3, "pin_l1": -0.5}, "n": 8}])
    put("depict_small_object", "gcr_alpha1",
        "pose(claw=front, grip=pinch, z=55) + move(claw, pitch, ±12, rep=2) + hold(t=0.8)",
        [{"op": "pose", "home": "1", "joints": _merge(A1["claw_front"], A1["grip_pinch"]), "n": 6},
         {"op": "osc", "joints": {"br1": 0.22}, "rep": 2, "n": 20, "home": "1",
          "base_joints": _merge(A1["claw_front"], A1["grip_pinch"])},
         {"op": "pose", "home": "1", "joints": _merge(A1["claw_front"], A1["grip_pinch"]), "n": 8}])

    # ---- depict_flying_bird ----
    put("depict_flying_bird", "gcr_alpha0",
        "move(hip1, yaw, ±30, rep=4) + move(hip2, yaw, ±30, rep=4)",
        [{"op": "osc", "joints": {"hip1": 0.5, "hip2": 0.5}, "rep": 4, "n": 36, "home": "0"}])
    put("depict_flying_bird", "gcr_alpha05",
        "pose(long=out, short=out) + move(long_pincer, yaw, ±35, rep=4) + move(short_pincer, yaw, ±35, rep=4)",
        [{"op": "pose", "home": "05", "joints": {"pin_r0": 0.9, "pin_l0": -0.9}, "n": 4},
         {"op": "osc", "joints": {"pin_r0": 0.55, "pin_l0": 0.55}, "rep": 4, "n": 36, "home": "05",
          "base_joints": {"pin_r0": 0.9, "pin_l0": -0.9}}])
    put("depict_flying_bird", "gcr_alpha1",
        "pose(paddle=out, claw=out) + move(paddle, yaw, ±35, rep=4) + move(claw, yaw, ±35, rep=4)",
        [{"op": "pose", "home": "1", "joints": A1["both_out"], "n": 4},
         {"op": "osc", "joints": {"bl0": 0.5, "br0": 0.5, "bl1": 0.25, "br1": 0.25}, "rep": 4, "n": 36,
          "home": "1", "base_joints": A1["both_out"]}])

    return cat


# ---------------------------------------------------------------------------
# plan executor
# ---------------------------------------------------------------------------

def _home_fn(tag):
    return {"0": home0, "05": home05, "1": home1}[tag]


def _traj(n):
    for i in range(n):
        yield i / max(n - 1, 1)


def _ease(t):
    """Smoothstep so pose/move links don't look linear-jerky at the ends."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_base(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _lerp_joints(a, b, t):
    keys = set(a) | set(b)
    out = {}
    for k in keys:
        av = a.get(k, 0.0)
        bv = b.get(k, av)
        out[k] = av + (bv - av) * t
    return out


def _n_approach(n):
    """Frames spent approaching a pose before holding (rest of n)."""
    if n <= 1:
        return n
    if n <= 8:
        return n  # short pose steps are pure approach into the next move
    return min(n, max(8, n // 2))


def execute_plan(plan):
    """Execute ops while carrying base/joint state so pose→pose / pose→move ease."""
    frames = []
    cur_base = (0.0, 0.0, 0.0)
    cur_joints = None

    def _seed(htag):
        nonlocal cur_joints
        if cur_joints is None:
            cur_joints = dict(_home_fn(htag)())
        return cur_joints

    for step in plan:
        op = step["op"]
        htag = step.get("home", "0")
        hfn = _home_fn(htag)
        _seed(htag)

        if op == "pose":
            target_base = step["base"] if "base" in step else cur_base
            target_joints = hfn(step.get("joints"))
            n = step.get("n", 10)
            n_move = _n_approach(n)
            for i in range(n_move):
                t = _ease(i / max(n_move - 1, 1))
                frames.append({
                    "base": _lerp_base(cur_base, target_base, t),
                    "joints": _lerp_joints(cur_joints, target_joints, t),
                })
            for _ in range(n - n_move):
                frames.append({"base": target_base, "joints": target_joints})
            cur_base, cur_joints = target_base, dict(target_joints)

        elif op == "lerp":
            to = step["to"]
            frm = step.get("from_joints")
            if frm is None:
                frm = {k: cur_joints.get(k, hfn().get(k, 0.0)) for k in to}
            target_base = step["base"] if "base" in step else cur_base
            target_full = None
            for t_raw in _traj(step.get("n", 12)):
                t = _ease(t_raw)
                j = {}
                keys = set(frm) | set(to)
                for k in keys:
                    a = frm.get(k, cur_joints.get(k, hfn().get(k, 0.0)))
                    b = to.get(k, a)
                    j[k] = a + (b - a) * t
                full = dict(cur_joints)
                full.update(j)
                base = _lerp_base(cur_base, target_base, t)
                frames.append({"base": base, "joints": full})
                target_full = full
            cur_base = target_base
            cur_joints = dict(target_full)

        elif op == "osc":
            amp = step.get("joints", {})
            center = step.get("center", {})
            base_j = step.get("base_joints", {})
            rep = step.get("rep", 2)
            n = step.get("n", 24)
            by = step.get("base_yaw")
            phase = step.get("phase", "full")
            # center of oscillation = current pose, then apply overrides
            root = dict(cur_joints)
            root.update(base_j)
            root.update(center)
            bx, by_, yaw0 = cur_base
            last = None
            for t in _traj(n):
                if phase == "half":
                    s = math.sin(math.pi * t)
                else:
                    s = math.sin(2 * math.pi * t * rep)
                j = dict(root)
                for k, a in amp.items():
                    base_v = root.get(k, 0.0)
                    j[k] = base_v + a * s
                yaw = yaw0 + ((by * s) if by is not None else 0.0)
                last = {"base": (bx, by_, yaw), "joints": j}
                frames.append(last)
            cur_base = last["base"]
            cur_joints = dict(last["joints"])

        elif op == "base_line":
            x = step["x"]
            n = step.get("n", 12)
            start = step["start_x"] if "start_x" in step else cur_base[0]
            end_x = start + x
            # joint target: override if given, else keep current
            if step.get("joints") is not None:
                target_joints = dict(cur_joints)
                target_joints.update(step["joints"])
            else:
                target_joints = dict(cur_joints)
            yaw = cur_base[2]
            y = cur_base[1]
            last = None
            for t_raw in _traj(n):
                t = _ease(t_raw)
                last = {
                    "base": (start + (end_x - start) * t, y, yaw),
                    "joints": _lerp_joints(cur_joints, target_joints, t),
                }
                frames.append(last)
            cur_base = last["base"]
            cur_joints = dict(last["joints"])

        elif op == "base_spin":
            turns = step.get("turns", 1)
            n = step.get("n", 24)
            if step.get("joints") is not None:
                target_joints = dict(cur_joints)
                target_joints.update(step["joints"])
            else:
                target_joints = dict(cur_joints)
            bx, by_, yaw0 = cur_base
            last = None
            for t_raw in _traj(n):
                t = _ease(t_raw)
                last = {
                    "base": (bx, by_, yaw0 + 2 * math.pi * turns * t),
                    "joints": _lerp_joints(cur_joints, target_joints, t),
                }
                frames.append(last)
            cur_base = last["base"]
            cur_joints = dict(last["joints"])

        elif op == "base_arc":
            r = step["r"]
            turns = step["turns"]
            n = step.get("n", 30)
            if step.get("joints") is not None:
                target_joints = dict(cur_joints)
                target_joints.update(step["joints"])
            else:
                target_joints = dict(cur_joints)
            last = None
            for t_raw in _traj(n):
                t = t_raw  # constant arc speed
                ang_local = 2 * math.pi * turns * t
                last = {
                    "base": (
                        cur_base[0] + r * math.cos(ang_local) - r,
                        cur_base[1] + r * math.sin(ang_local),
                        cur_base[2] + ang_local,
                    ),
                    "joints": _lerp_joints(cur_joints, target_joints, _ease(t)),
                }
                frames.append(last)
            for _ in range(step.get("hold_end", 0)):
                frames.append({"base": last["base"], "joints": last["joints"]})
            cur_base = last["base"]
            cur_joints = dict(last["joints"])
        else:
            raise ValueError(op)
    return frames


# ---------------------------------------------------------------------------
# render / html / yaml
# ---------------------------------------------------------------------------

def _font(size=13):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _quat_yaw(yaw):
    h = 0.5 * yaw
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=np.float64)


def _set_joint(data, model, name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = value


def _set_base(data, model, x, y, yaw, z):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = [x, y, z]
    data.qpos[adr + 3:adr + 7] = _quat_yaw(yaw)


def _overlay(rgb, title, subtitle):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 42], fill=(255, 255, 255, 230))
    draw.text((6, 3), title, fill=(15, 15, 20), font=_font(12))
    draw.text((6, 22), subtitle[:70], fill=(50, 55, 65), font=_font(10))
    return np.asarray(img.convert("RGB"))


def render_clip(model, frames, out_gif, title, subtitle, lookat, distance, base_z):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = 180
    cam.elevation = -8
    imgs = []
    for cfg in frames:
        mujoco.mj_resetData(model, data)
        x, y, yaw = cfg.get("base", (0.0, 0.0, 0.0))
        _set_base(data, model, x, y, yaw, base_z)
        for jn, v in cfg.get("joints", {}).items():
            _set_joint(data, model, jn, v)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        imgs.append(_overlay(renderer.render(), title, subtitle))
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, imgs, fps=FPS, loop=0)
    renderer.close()


def write_yaml(catalog):
    doc = {
        "meta": {
            "version": "0.1",
            "family": "oddkin",
            "n_cues": len(CUES),
            "morphologies": ["gcr_alpha0", "gcr_alpha05", "gcr_alpha1"],
            "fewshot_reference": [
                "raising_hand_greeting", "wave_hi", "bow_greeting",
                "blow_kiss", "circle_temple_crazy",
            ],
            "note": "Rematerialized Motion Codes for the paper's 30-cue pool.",
        },
        "cues": {},
    }
    for cue in CUES:
        doc["cues"][cue] = {
            morph: catalog[cue][morph]["code"] for morph in ("gcr_alpha0", "gcr_alpha05", "gcr_alpha1")
        }
    YML_OUT.write_text(yaml.safe_dump(doc, sort_keys=False, width=100, allow_unicode=True))
    print("wrote", YML_OUT)


def write_html(catalog, out_html):
    rows = []
    for cue in CUES:
        cells = []
        for morph, label, *_ in MORPHS:
            code = catalog[cue][morph]["code"]
            cells.append(f"""
        <td><figure class="cell">
          <img src="{morph}/{cue}.gif" alt="{morph}-{cue}"/>
          <figcaption><div class="m">{label}</div><code>{code}</code></figcaption>
        </figure></td>""")
        rows.append(f"<tr><th><div class='cue'>{cue}</div></th>{''.join(cells)}</tr>")
    out_html.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Oddkin · 30 cues × 3</title>
<style>
:root {{ --bg:#0c1017; --panel:#151a24; --text:#e9eef7; --muted:#9aa6b8; --accent:#7dd3c0; --line:#2a3344; }}
body {{ margin:0; padding:24px; font-family:"IBM Plex Sans",sans-serif; background:var(--bg); color:var(--text); }}
h1 {{ margin:0 0 6px; font-size:1.35rem; }}
.lede {{ color:var(--muted); max-width:95ch; margin-bottom:18px; line-height:1.45; font-size:.92rem; }}
table {{ width:100%; border-collapse:separate; border-spacing:10px; }}
th {{ text-align:left; vertical-align:top; width:150px; }}
.cue {{ font-size:.8rem; color:var(--accent); font-family:ui-monospace,monospace; word-break:break-word; }}
td {{ vertical-align:top; width:28%; }}
.cell {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.cell img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#000; }}
figcaption {{ padding:7px 8px 9px; }}
.m {{ font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; }}
code {{ font-size:.62rem; color:#c8f0e6; word-break:break-word; line-height:1.3; }}
</style></head><body>
<h1>Oddkin · 30 cues × 3 morphologies</h1>
<p class="lede">Paper cue pool rematerialized with Oddkin attributes, styled after the 5 few-shot patterns
(raise / wave / bow / kiss / circle). Source: <code>yml/cues30_oddkin.yaml</code>.</p>
<table><thead><tr><th>cue</th><th>α=0</th><th>α=0.5</th><th>α=1.0</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>""")
    print("wrote", out_html)


def main():
    catalog = build_catalog()
    assert set(catalog) == set(CUES)
    write_yaml(catalog)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for morph, *_ in MORPHS:
        models[morph] = mujoco.MjModel.from_xml_path(str(XML_DIR / f"{morph}.xml"))

    for morph, label, lookat, dist, base_z in MORPHS:
        print("rendering", morph, "...")
        out_sub = OUT_DIR / morph
        out_sub.mkdir(parents=True, exist_ok=True)
        for cue in CUES:
            entry = catalog[cue][morph]
            frames = execute_plan(entry["plan"])
            render_clip(
                models[morph], frames, out_sub / f"{cue}.gif",
                title=f"{label} · {cue}", subtitle=entry["code"],
                lookat=lookat, distance=dist, base_z=base_z,
            )
        print(" ", len(CUES), "cues")

    write_html(catalog, OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
