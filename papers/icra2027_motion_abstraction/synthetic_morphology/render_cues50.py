#!/usr/bin/env python3
"""Extend the Oddkin gallery from the paper 30 to all 50 evaluation cues."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

# The original renderer uses imageio only to write GIFs.  Keep this renderer
# runnable in the project MuJoCo Python (which has Pillow but not imageio).
imageio = types.ModuleType("imageio")
v2 = types.ModuleType("imageio.v2")
def _mimsave(path, frames, fps=11, loop=0):
    imgs = [Image.fromarray(np.asarray(frame).astype("uint8")) for frame in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=loop)
v2.mimsave = _mimsave
imageio.v2 = v2
sys.modules.setdefault("imageio", imageio)
sys.modules.setdefault("imageio.v2", v2)

import render_cues30 as r  # noqa: E402


EXTRA = [
    "self_scratch_head_confused", "pat_stomach", "fan_face_hot", "cover_mouth_gasp",
    "fistbump_invite", "throat_cut_stop", "visor_search", "yield_turn", "request_turn",
    "slow_down_request_palm_down", "scale_expand", "clarification_request_palm_tilt",
    "continue_on_forward_roll", "interrupt_hard", "interrupt_soft", "self_hug",
    "push_pull_dual", "wait_for_me_backward_palm", "frustration_down_throw", "clap_applause",
]


def _entry(code, plan):
    return {"code": code, "plan": plan}


def alpha0(cue):
    leg = {"hip0": 0.55, "knee0": -0.45}
    if cue in {"self_scratch_head_confused", "pat_stomach", "fan_face_hot", "cover_mouth_gasp", "visor_search"}:
        return _entry("pose(leg0=present) + move(hip0, yaw, ±25, rep=3)", [{"op":"pose","home":"0","joints":leg,"n":8},{"op":"osc","joints":{"hip0":0.35},"rep":3,"n":30,"home":"0","base_joints":leg}])
    if cue in {"fistbump_invite", "request_turn", "continue_on_forward_roll"}:
        return _entry("pose(leg0=present) + path(base, line, x=+0.18)", [{"op":"pose","home":"0","joints":leg,"n":8},{"op":"base_line","x":0.18,"n":18}])
    if cue in {"yield_turn", "wait_for_me_backward_palm"}:
        return _entry("pose(leg0=present) + path(base, line, x=-0.18)", [{"op":"pose","home":"0","joints":leg,"n":8},{"op":"base_line","x":-0.18,"n":18}])
    if cue in {"throat_cut_stop", "interrupt_hard", "interrupt_soft", "slow_down_request_palm_down"}:
        hold = 16 if cue != "interrupt_hard" else 8
        return _entry("pose(leg0=raise, crown=quarter) + hold(t=1.0)", [{"op":"pose","home":"0","joints":{"hip0":0.8,"knee0":-0.2,"crown_spin":1.1},"n":hold}])
    if cue in {"scale_expand", "self_hug", "push_pull_dual", "clap_applause"}:
        return _entry("move(hip0, yaw, ±30, rep=2) + move(hip1, yaw, ∓30, rep=2)", [{"op":"osc","joints":{"hip0":0.45,"hip1":-0.45},"rep":2,"n":30,"home":"0"}])
    if cue == "clarification_request_palm_tilt":
        return _entry("path(crown, arc, r=0, deg=±40, rep=2)", [{"op":"osc","joints":{"crown_spin":0.65},"rep":2,"n":28,"home":"0"}])
    return _entry("move(crown, yaw, ±45, rep=2) + path(base, arc, r=0, deg=90)", [{"op":"osc","joints":{"crown_spin":0.7},"rep":2,"n":26,"home":"0"},{"op":"base_spin","turns":0.25,"n":16}])


def alpha05(cue):
    long = {"pin_r0": 0.15, "pin_r1": -0.65}
    short = {"pin_l0": -0.65, "pin_l1": -0.25}
    if cue in {"self_scratch_head_confused", "pat_stomach", "fan_face_hot", "cover_mouth_gasp", "visor_search"}:
        return _entry("pose(long=in, short=still) + move(long_pincer, pitch, ±20, rep=3)", [{"op":"pose","home":"05","joints":long,"n":8},{"op":"osc","joints":{"pin_r1":0.38},"rep":3,"n":30,"home":"05","base_joints":long}])
    if cue in {"fistbump_invite", "request_turn", "continue_on_forward_roll"}:
        return _entry("pose(long=front) + path(base, line, x=+0.18)", [{"op":"pose","home":"05","joints":long,"n":8},{"op":"base_line","x":0.18,"n":18}])
    if cue in {"yield_turn", "wait_for_me_backward_palm"}:
        return _entry("pose(long=out) + path(base, line, x=-0.18)", [{"op":"pose","home":"05","joints":long,"n":8},{"op":"base_line","x":-0.18,"n":18}])
    if cue in {"throat_cut_stop", "interrupt_hard", "interrupt_soft", "slow_down_request_palm_down"}:
        return _entry("pose(long=front) + move(collar, yaw, ±12, rep=2) + hold(t=1.0)", [{"op":"pose","home":"05","joints":long,"n":8},{"op":"osc","joints":{"collar_spin":0.25},"rep":2,"n":20,"home":"05","base_joints":long}])
    if cue in {"scale_expand", "self_hug", "push_pull_dual", "clap_applause"}:
        return _entry("pose(long=out, short=out) + move(long_pincer, yaw, ±25, rep=2)", [{"op":"pose","home":"05","joints":r._merge(long,short),"n":8},{"op":"osc","joints":{"pin_r0":0.42,"pin_l0":-0.42},"rep":2,"n":30,"home":"05","base_joints":r._merge(long,short)}])
    if cue == "clarification_request_palm_tilt":
        return _entry("path(torso, arc, r=0, deg=±30, rep=2)", [{"op":"osc","joints":{"v1_yaw":0.35,"v2_twist":0.28},"rep":2,"n":28,"home":"05"}])
    return _entry("pose(torso=high) + move(long_pincer, pitch, -35) + path(base, arc, r=0, deg=90)", [{"op":"pose","home":"05","joints":r.SP05["high"],"n":8},{"op":"lerp","home":"05","to":{"pin_r1":-0.9},"n":14},{"op":"base_spin","turns":0.25,"n":16}])


def alpha1(cue):
    paddle, claw = r.A1["paddle_in"], r.A1["claw_in"]
    if cue in {"self_scratch_head_confused", "pat_stomach", "fan_face_hot", "cover_mouth_gasp", "visor_search"}:
        return _entry("pose(claw=in, head=center) + move(claw, pitch, ±20, rep=3)", [{"op":"pose","home":"1","joints":claw,"n":8},{"op":"osc","joints":{"br1":0.35},"rep":3,"n":30,"home":"1","base_joints":claw}])
    if cue in {"fistbump_invite", "request_turn", "continue_on_forward_roll"}:
        return _entry("pose(claw=front, grip=close) + path(base, line, x=+0.18)", [{"op":"pose","home":"1","joints":r._merge(r.A1["claw_front"],r.A1["grip_close"]),"n":8},{"op":"base_line","x":0.18,"n":18}])
    if cue in {"yield_turn", "wait_for_me_backward_palm"}:
        return _entry("pose(paddle=out, palm=palmup) + path(base, line, x=-0.18)", [{"op":"pose","home":"1","joints":r.A1["paddle_out"],"n":8},{"op":"base_line","x":-0.18,"n":18}])
    if cue in {"throat_cut_stop", "interrupt_hard", "interrupt_soft", "slow_down_request_palm_down"}:
        palm = r.A1["paddle_front"]
        return _entry("pose(paddle=front, palm=palmfront) + hold(t=1.0)", [{"op":"pose","home":"1","joints":palm,"n":16 if cue != "interrupt_hard" else 8}])
    if cue in {"scale_expand", "self_hug", "push_pull_dual", "clap_applause"}:
        both = r.A1["both_out"] if cue == "scale_expand" else r._merge(paddle,claw)
        return _entry("pose(paddle=out, claw=out) + move(paddle, z, ±20, rep=2) + move(claw, z, ∓20, rep=2)", [{"op":"pose","home":"1","joints":both,"n":8},{"op":"osc","joints":{"bl1":0.35,"br1":-0.35},"rep":2,"n":30,"home":"1","base_joints":both}])
    if cue == "clarification_request_palm_tilt":
        return _entry("pose(paddle=front, palm=palmup) + move(paddle, roll, ±25, rep=2)", [{"op":"pose","home":"1","joints":r.A1["paddle_front"],"n":8},{"op":"osc","joints":{"bl2":0.42},"rep":2,"n":28,"home":"1","base_joints":r.A1["paddle_front"]}])
    return _entry("pose(torso=high, claw=down) + move(claw, z, -35) + path(base, arc, r=0, deg=90)", [{"op":"pose","home":"1","joints":r._merge(r.A1["claw_down"] if "claw_down" in r.A1 else {},{"trunk1_pitch":-0.35}),"n":8},{"op":"lerp","home":"1","to":{"br1":0.55},"n":14},{"op":"base_spin","turns":0.25,"n":16}])


def main():
    r.CUES = r.CUES + EXTRA
    r.YML_OUT = r.ROOT / "yml" / "cues50_oddkin.yaml"
    r.OUT_DIR = r.ROOT / "renders" / "cues50"
    catalog = r.build_catalog()
    for cue in EXTRA:
        catalog[cue] = {"gcr_alpha0": alpha0(cue), "gcr_alpha05": alpha05(cue), "gcr_alpha1": alpha1(cue)}
    r.write_yaml(catalog)
    r.OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {m: r.mujoco.MjModel.from_xml_path(str(r.XML_DIR / f"{m}.xml")) for m, *_ in r.MORPHS}
    for morph, label, lookat, distance, base_z in r.MORPHS:
        print("rendering", morph, flush=True)
        for cue in r.CUES:
            entry = catalog[cue][morph]
            r.render_clip(models[morph], r.execute_plan(entry["plan"]), r.OUT_DIR / morph / f"{cue}.gif", f"{label} · {cue}", entry["code"], lookat, distance, base_z)
    out_html = r.OUT_DIR / "index.html"
    r.write_html(catalog, out_html)
    text = out_html.read_text().replace("Oddkin · 30 cues × 3", "Oddkin · 50 cues × 3").replace("paper's 30-cue pool", "50-cue evaluation pool").replace("cues30_oddkin.yaml", "cues50_oddkin.yaml")
    out_html.write_text(text)


if __name__ == "__main__":
    main()
