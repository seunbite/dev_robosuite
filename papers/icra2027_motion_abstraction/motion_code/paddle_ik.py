#!/usr/bin/env python3
"""Jacobian IK: put paddlebot claw (and paddle) on body landmarks.

Solves damped least-squares IK so ``ee`` (claw tip) reaches cue-specific
landmarks (mouth, face, chest, abdomen, …). Mirror cues also IK the paddle tip.

Outputs:
  bank/_paddle_xyz_iter/ik_body/joints.json   — solved joint dicts
  bank/_paddle_xyz_iter/ik_body/index.html    — still gallery
  data/.../paddle_ik_joints.json              — consumed by render_paddlebot90
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ODDKIN = PAPER / "synthetic_morphology"
sys.path[:0] = [str(MC), str(ODDKIN), str(PAPER), str(REPO)]

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import render_cues30 as rc  # noqa: E402
import render_paddlebot90 as pb  # noqa: E402

OUT_DIR = MC / "bank/_paddle_xyz_iter/ik_body"
OUT_JOINTS = OUT_DIR / "joints.json"
OUT_CFG = OUT_DIR / "paddle_ik_joints.json"  # paper-local; render_paddlebot90 also reads here
XML = ODDKIN / "xml" / "gcr_alpha1.xml"
BASE_Z = 0.13

CLAW_JOINTS = ("br0", "br1", "br2", "br3")
PADDLE_JOINTS = ("bl0", "bl1", "bl2")
# Trunk stays upright for most cues — IK is arm-only.
TRUNK_ASSIST: tuple[str, ...] = ()
# pat_stomach only: light pitch assist (arm cannot reach abdomen upright).
TRUNK_STOMACH: tuple[str, ...] = ("trunk1_pitch", "trunk2_pitch", "trunk3_pitch")
TRUNK_STOMACH_CAP = 0.42  # rad; arm-only cannot reach belly upright

# cue → landmark key (+ mirror paddle to same/pair landmark)
# Targets are offsets from named body frames (meters, robot frame).
LANDMARK_OFFSETS: dict[str, tuple[str, np.ndarray]] = {
    "mouth": ("head", np.array([0.08, 0.00, -0.055])),
    "face": ("head", np.array([0.10, 0.00, -0.01])),
    "eye": ("head", np.array([0.07, -0.05, 0.01])),
    "temple": ("head", np.array([0.04, -0.10, 0.01])),
    "ear_r": ("head", np.array([0.02, -0.12, 0.00])),
    "ear_l": ("head", np.array([0.02, 0.12, 0.00])),
    "forehead": ("head", np.array([0.07, 0.00, 0.05])),
    "chin": ("head", np.array([0.08, 0.00, -0.10])),
    # lower belly on trunk1 (needs slight trunk pitch to reach)
    "abdomen": ("trunk1", np.array([0.16, -0.03, 0.06])),
    "chest": ("trunk2", np.array([0.16, -0.03, 0.02])),
    "heart": ("trunk2", np.array([0.16, 0.03, 0.00])),
    "throat": ("trunk3", np.array([0.13, 0.00, 0.02])),
    "back_r": ("trunk3", np.array([-0.10, -0.14, 0.02])),
    "beside_face": ("head", np.array([0.08, -0.14, 0.00])),
    "shoulder_r": ("trunk3", np.array([0.06, -0.18, 0.04])),
    "cross_chest": ("trunk2", np.array([0.15, 0.00, 0.00])),
    "clap": ("trunk2", np.array([0.17, 0.00, 0.05])),
    "upper_arm_r": ("trunk3", np.array([0.08, -0.14, -0.02])),
    "upper_arm_l": ("trunk3", np.array([0.08, 0.14, -0.02])),
}

# Body-anchored cues → (claw landmark, paddle landmark|None)
CUE_LANDMARKS: dict[str, tuple[str, str | None]] = {
    "cover_yawn_sleepy": ("mouth", None),
    "rub_eye_tired": ("eye", None),
    "salute": ("temple", None),  # same site as self_scratch; joints copied below
    "facepalm": ("face", None),
    "self_scratch_head_confused": ("temple", None),
    "fan_face_hot": ("beside_face", None),
    "tap_temple_think": ("temple", None),
    "circle_temple_crazy": ("temple", None),
    "cover_mouth_gasp": ("mouth", None),
    "shh_gesture": ("mouth", None),
    "visor_search": ("forehead", None),
    "chin_flick_dismiss": ("chin", None),
    "cover_ears_too_loud": ("ear_r", "ear_l"),
    "point_self": ("chest", None),
    "flex_bicep": ("mouth", None),  # same site as shh; joints copied below
    "cross_arms_defensive": ("cross_chest", "cross_chest"),
    "palms_together_thank_you": ("clap", "clap"),
    "pat_stomach": ("abdomen", None),
    "throat_cut_stop": ("throat", None),
    "pat_self_back": ("back_r", None),
    "hand_over_heart_sincerity": ("heart", None),
    "self_hug": ("cross_chest", "cross_chest"),
    "rub_arms_cold": ("upper_arm_r", "upper_arm_l"),
    "clap_applause": ("clap", "clap"),
}

# After IK, force-identical poses (user preference).
POSE_COPY: dict[str, str] = {
    "salute": "self_scratch_head_confused",
    "flex_bicep": "shh_gesture",
}


def _font(size: int = 13) -> ImageFont.ImageFont:
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _joint_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
    out = []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise KeyError(n)
        out.append(int(jid))
    return out


def _apply_joints(model: mujoco.MjModel, data: mujoco.MjData, joints: dict[str, float]) -> None:
    mujoco.mj_resetData(model, data)
    rc._set_base(data, model, 0.0, 0.0, 0.0, BASE_Z)
    for jn, v in joints.items():
        try:
            rc._set_joint(data, model, jn, float(v))
        except Exception:
            pass
    mujoco.mj_forward(model, data)


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def _landmark_world(model: mujoco.MjModel, data: mujoco.MjData, key: str) -> np.ndarray:
    body, off = LANDMARK_OFFSETS[key]
    return _body_pos(model, data, body) + off


def _read_joints(model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        out[n] = float(data.qpos[model.jnt_qposadr[jid]])
    return out


def _clamp_joint(model: mujoco.MjModel, jid: int, q: float) -> float:
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        return float(np.clip(q, lo, hi))
    return float(q)


def ik_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_fn,
    joint_names: tuple[str, ...],
    mode: str = "site",
    site_or_body: str = "ee",
    n_iter: int = 160,
    damping: float = 1.5e-2,
    step: float = 0.55,
    tol: float = 0.015,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Damped LS IK. ``target_fn()`` returns a moving body-relative target each iter."""
    jids = _joint_ids(model, joint_names)
    dof_idxs = [int(model.jnt_dofadr[j]) for j in jids]
    if mode == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_or_body)
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_or_body)

    err_norm = 1e9
    cur = np.zeros(3)
    target = np.zeros(3)
    for _ in range(n_iter):
        mujoco.mj_forward(model, data)
        target = np.asarray(target_fn(), dtype=float)
        if mode == "site":
            cur = data.site_xpos[sid].copy()
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        else:
            cur = data.xpos[bid].copy()
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, cur, bid)
        err = target - cur
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break
        J = jacp[:, dof_idxs]
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)
        for k, jid in enumerate(jids):
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = _clamp_joint(model, jid, data.qpos[adr] + step * float(dq[k]))
    mujoco.mj_forward(model, data)
    if mode == "site":
        cur = data.site_xpos[sid].copy()
    else:
        cur = data.xpos[bid].copy()
    target = np.asarray(target_fn(), dtype=float)
    return float(np.linalg.norm(target - cur)), cur, target


def _seed_joints(dir_name: str, *, mirror: bool) -> dict[str, float]:
    j = dict(rc.home1())
    j.update(rc.A1["grip_open"])
    claw_key = f"claw_{dir_name}" if f"claw_{dir_name}" in pb.A1_DIRS else "claw_front"
    j.update(pb._preset(claw_key))
    if mirror:
        pad_key = f"paddle_{dir_name}" if f"paddle_{dir_name}" in pb.A1_DIRS else "paddle_front"
        j.update(pb._preset(pad_key))
    else:
        j.update(pb._preset("paddle_still"))
    return j


def solve_cue(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cue: str,
) -> dict[str, Any]:
    claw_lm, pad_lm = CUE_LANDMARKS[cue]
    fields = pb._parse_pose_fields(pb.CANVAS[cue]["pose"])
    arm_dir = fields.get("dir") or "front"
    if "|" in arm_dir:
        arm_dir = arm_dir.split("|")[0]
    mirror = (fields.get("left_arm") or "still") in ("mirror", "opposed") or pad_lm is not None

    seeds = [arm_dir, "in", "front", "up", "back", "down"]
    if cue == "pat_stomach":
        seeds = ["back", "front", "in", "down", "up"]
    seen: set[str] = set()
    best: dict[str, Any] | None = None

    use_trunk = cue == "pat_stomach"
    claw_dofs: tuple[str, ...] = CLAW_JOINTS + (TRUNK_STOMACH if use_trunk else ())

    def claw_target():
        t = _landmark_world(model, data, claw_lm)
        # slight lateral bias for dual-hand meet so arms don't fight
        if pad_lm is not None and claw_lm in ("clap", "cross_chest"):
            t = t + np.array([0.0, -0.04, 0.00])
        return t

    def pad_target():
        t = _landmark_world(model, data, pad_lm)  # type: ignore[arg-type]
        if claw_lm in ("clap", "cross_chest"):
            t = t + np.array([0.0, 0.04, 0.00])
        return t

    for seed in seeds:
        if seed in seen or seed not in ("front", "back", "in", "out", "up", "down"):
            continue
        seen.add(seed)
        joints = _seed_joints(seed, mirror=mirror)
        _apply_joints(model, data, joints)

        err_c, pos_c, tgt_claw = ik_point(
            model,
            data,
            target_fn=claw_target,
            joint_names=claw_dofs,
            mode="site",
            site_or_body="ee",
            n_iter=200 if use_trunk else 160,
        )
        err_p = 0.0
        if pad_lm is not None:
            for _ in range(3):
                err_p, _, _ = ik_point(
                    model,
                    data,
                    target_fn=pad_target,
                    joint_names=PADDLE_JOINTS,
                    mode="body",
                    site_or_body="branch_l2",
                    n_iter=80,
                )
                err_c, pos_c, tgt_claw = ik_point(
                    model,
                    data,
                    target_fn=claw_target,
                    joint_names=CLAW_JOINTS,
                    mode="site",
                    site_or_body="ee",
                    n_iter=80,
                )

        solved = _read_joints(
            model,
            data,
            CLAW_JOINTS
            + PADDLE_JOINTS
            + TRUNK_STOMACH
            + (
                "claw0",
                "claw1",
                "claw2",
                "neck_yaw",
                "neck_pitch",
                "head_roll",
                "trunk1_yaw",
                "trunk1_pitch",
                "trunk2_yaw",
                "trunk2_pitch",
                "trunk3_pitch",
                "trunk3_roll",
                "sat_spin",
            ),
        )
        # Keep torso upright except mild pitch for stomach reach.
        for tn in ("trunk1_yaw", "trunk2_yaw", "trunk3_roll", "sat_spin"):
            solved[tn] = 0.0
        if use_trunk:
            for tn in TRUNK_STOMACH:
                solved[tn] = float(np.clip(solved.get(tn, 0.0), -TRUNK_STOMACH_CAP, TRUNK_STOMACH_CAP))
            # re-FK after clamp so reported ee matches joints
            _apply_joints(model, data, solved)
            mujoco.mj_forward(model, data)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
            pos_c = data.site_xpos[sid].copy()
            tgt_claw = claw_target()
            err_c = float(np.linalg.norm(tgt_claw - pos_c))
        else:
            for tn in TRUNK_STOMACH:
                solved[tn] = 0.0
        solved.update(rc.A1["grip_open"])

        trunk_pen = sum(abs(solved.get(tn, 0.0)) for tn in TRUNK_STOMACH) if use_trunk else 0.0
        score = err_c + (0.8 * err_p if pad_lm else 0.0) + 0.25 * trunk_pen
        cand = {
            "cue": cue,
            "claw_landmark": claw_lm,
            "paddle_landmark": pad_lm,
            "seed": seed,
            "err_claw": round(err_c, 4),
            "err_paddle": round(err_p, 4) if pad_lm else None,
            "score": round(score, 4),
            "ee": [round(float(x), 3) for x in pos_c],
            "target_claw": [round(float(x), 3) for x in tgt_claw],
            "joints": {k: round(float(v), 4) for k, v in solved.items()},
        }
        if best is None or score < best["score"]:
            best = cand
        if score < 0.03:
            break

    assert best is not None
    return best


def render_still(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    joints: dict[str, float],
) -> Image.Image:
    data = mujoco.MjData(model)
    _apply_joints(model, data, joints)
    # draw small spheres at ee via overlay text only
    renderer.update_scene(data, camera=cam)
    return Image.fromarray(renderer.render()).convert("RGB")


def run(*, cues: list[str] | None = None, force: bool = False) -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pb.OUT_CFG.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    wanted = cues or list(CUE_LANDMARKS)
    results: list[dict[str, Any]] = []

    for i, cue in enumerate(wanted, 1):
        if cue not in CUE_LANDMARKS:
            raise KeyError(cue)
        sol = solve_cue(model, data, cue)
        results.append(sol)
        print(
            f"[{i}/{len(wanted)}] {cue}: seed={sol['seed']} "
            f"err_claw={sol['err_claw']:.3f}m "
            f"err_pad={sol['err_paddle']} "
            f"lm={sol['claw_landmark']}",
            flush=True,
        )

    # Force pose aliases (salute≡scratch, flex≡shh).
    by_cue = {r["cue"]: r for r in results}
    for dst, src in POSE_COPY.items():
        if dst in by_cue and src in by_cue:
            src_r = by_cue[src]
            dst_r = by_cue[dst]
            dst_r["joints"] = dict(src_r["joints"])
            dst_r["claw_landmark"] = src_r["claw_landmark"]
            dst_r["paddle_landmark"] = src_r["paddle_landmark"]
            dst_r["err_claw"] = src_r["err_claw"]
            dst_r["err_paddle"] = src_r["err_paddle"]
            dst_r["seed"] = f"copy:{src}"
            dst_r["score"] = src_r["score"]
            dst_r["ee"] = list(src_r["ee"])
            dst_r["target_claw"] = list(src_r["target_claw"])
            print(f"[copy] {dst} ← {src}", flush=True)

    payload = {"platform": "gcr_alpha1", "method": "jacobian_ik", "cues": {r["cue"]: r for r in results}}
    OUT_JOINTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # slim file for renderer (paper-local + repo configs)
    slim = {
        "platform": "gcr_alpha1",
        "cues": {r["cue"]: r["joints"] for r in results},
        "meta": {
            r["cue"]: {
                "claw_landmark": r["claw_landmark"],
                "paddle_landmark": r["paddle_landmark"],
                "err_claw": r["err_claw"],
                "err_paddle": r["err_paddle"],
            }
            for r in results
        },
    }
    local_slim = OUT_DIR / "paddle_ik_joints.json"
    local_slim.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"[ik] joints → {local_slim}", flush=True)

    # gallery
    renderer = mujoco.Renderer(model, height=rc.H, width=rc.W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.40)
    cam.distance = 2.05
    cam.azimuth = 180
    cam.elevation = -8
    cards = []
    try:
        for r in results:
            img = render_still(model, renderer, cam, r["joints"])
            # annotate
            bar = 36
            out = Image.new("RGB", (img.width, img.height + bar), "white")
            out.paste(img, (0, bar))
            draw = ImageDraw.Draw(out)
            err_p = r["err_paddle"]
            msg = f"{r['cue']}  lm={r['claw_landmark']}  err={r['err_claw']:.3f}m"
            if err_p is not None:
                msg += f"  pad_err={err_p:.3f}m"
            draw.text((6, 8), msg, fill=(20, 20, 20), font=_font(12))
            png = OUT_DIR / f"{r['cue']}.png"
            out.save(png)
            cards.append(
                "<article>"
                f"<h2>{html.escape(r['cue'])}</h2>"
                f"<p class='meta'>claw→<b>{html.escape(r['claw_landmark'])}</b> "
                f"err={r['err_claw']:.3f}m"
                + (
                    f" · paddle→<b>{html.escape(str(r['paddle_landmark']))}</b> err={r['err_paddle']:.3f}m"
                    if r["paddle_landmark"]
                    else ""
                )
                + f" · seed={html.escape(r['seed'])}</p>"
                f"<img src='{html.escape(r['cue']+'.png')}?v=3'>"
                "</article>"
            )
    finally:
        renderer.close()

    (OUT_DIR / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Paddle IK body landmarks</title>"
        "<style>body{font-family:system-ui;margin:16px;background:#e8eaef}"
        "h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}"
        "article{background:#fff;padding:10px;border-radius:10px;border:1px solid #ccd}"
        "img{width:100%} .meta{font-size:12px;color:#444}</style>"
        "<h1>Paddlebot IK → body landmarks</h1>"
        "<p>Claw <code>ee</code> IK’d onto mouth/face/chest/… "
        "(paddle tip too when mirrored).</p>"
        f"<div class='grid'>{''.join(cards)}</div>",
        encoding="utf-8",
    )
    print(OUT_DIR / "index.html", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", default="", help="comma-separated subset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cues = [c.strip() for c in args.cues.split(",") if c.strip()] or None
    run(cues=cues, force=args.force)


if __name__ == "__main__":
    main()
