"""Render helpers: pose tiles, alpha trajectories, GIF lookup."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM, _REPO / "adhoc" / "vlm_test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verify_pose_vlm import (  # noqa: E402
    _crop_tile_from_group,
    _first_pose,
    _load_tile_pick,
)

ROBOT = "IIWA"
HZ = 10
POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
CACHE = _REPO / "data/results/paper_figures/cache"


def repo() -> Path:
    return _REPO


def load_pose_cfg() -> list[dict[str, Any]]:
    return json.loads(POSE_CFG.read_text(encoding="utf-8"))


def cfg_by_idx() -> dict[int, dict[str, Any]]:
    return {int(r["idx"]): r for r in load_pose_cfg()}


def manifest90_rows() -> list[dict[str, Any]]:
    from pilot90_experiment_suite import manifest90_rows_from_cfg

    return manifest90_rows_from_cfg(load_pose_cfg())


def parse_idx_arg(idx_arg: str, rows: list[dict[str, Any]]) -> list[int]:
    if idx_arg.strip().lower() == "all":
        return [int(r["idx"]) for r in sorted(rows, key=lambda r: int(r["idx"]))]
    out: list[int] = []
    for part in idx_arg.replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _pose_gt_by_cue() -> dict[str, str]:
    return {
        str(r["cue"]): str(r["groundtruth"])
        for r in json.loads(
            (_REPO / "data/results/verify/pilot40_pose_eval_consolidated.json").read_text(encoding="utf-8")
        ).get("rows", [])
        if r.get("groundtruth")
    }


def pose_correct_idxs() -> list[int]:
    """Cue idx where generation config matches human pose GT (any-pose rule, exp1)."""
    from pilot90_experiment_suite import pose_generation_correct_any

    gt = _pose_gt_by_cue()
    ok: list[int] = []
    for row in manifest90_rows():
        cue = str(row.get("cue", ""))
        if cue not in gt:
            continue
        if pose_generation_correct_any(row, gt[cue]) is True:
            ok.append(int(row["idx"]))
    return sorted(ok)


def pose_split_idxs(idxs: list[int]) -> tuple[list[int], list[int]]:
    """Partition idx into pose-GT-correct (top) vs incorrect (bottom)."""
    from pilot90_experiment_suite import pose_generation_correct_any

    gt = _pose_gt_by_cue()
    by_idx = cfg_by_idx()
    correct: list[int] = []
    wrong: list[int] = []
    for idx in idxs:
        row = by_idx.get(idx)
        if not row:
            continue
        cue = str(row.get("cue", ""))
        if cue not in gt:
            continue
        if pose_generation_correct_any(row, gt[cue]) is True:
            correct.append(idx)
        else:
            wrong.append(idx)
    return correct, wrong


def motion_correct_idxs() -> list[int]:
    """Cue idx where generation tail matches motion component GT (exp7 rule)."""
    from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps

    ann = {
        int(a["cue_idx"]): a
        for a in json.loads(
            (_REPO / "data/results/verify/pilot40_motion_component_gt.json").read_text(encoding="utf-8")
        ).get("annotations", [])
    }
    ok: list[int] = []
    for row in manifest90_rows():
        idx = int(row["idx"])
        entry = ann.get(idx) or {}
        comp = entry.get("component")
        raw = (entry.get("annotation_raw") or "").strip()
        if not comp and not entry.get("always_correct") and raw.lower() != "none":
            continue
        tail = _tail_steps(row.get("movements") or [])
        if entry.get("always_correct") or (comp and comp.get("kind") == "any"):
            match = True
        elif comp:
            match, _ = _tail_matches_component(tail, comp)
        else:
            continue
        if match:
            ok.append(idx)
    return sorted(ok)


def motion_split_idxs(idxs: list[int]) -> tuple[list[int], list[int]]:
    """Partition idx into motion-GT-correct (top) vs incorrect (bottom)."""
    ok = set(motion_correct_idxs())
    correct = [i for i in idxs if i in ok]
    wrong = [i for i in idxs if i not in ok]
    return correct, wrong


def pick_gif(cue: str) -> Path | None:
    from motion_media_paths import pick_latest_gif

    return pick_latest_gif(_REPO, cue, pilot90=True)


def pose_png_for_row(row: dict[str, Any], out_png: Path) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file():
        return out_png
    pose = _first_pose(row)
    d, g = pose.get("dir"), pose.get("gripper_orientation")
    if not d or not g:
        return None
    tile_pick = _load_tile_pick(TILE_PICK)
    idx = tile_pick.get((d, g))
    if idx is None:
        return None
    group_path = TILE_DIR / f"group_{d}_{g}.png"
    if not group_path.is_file():
        return None
    cell = _crop_tile_from_group(Image.open(group_path).convert("RGB"), idx)
    cell.save(out_png)
    return out_png


def _sample_for_cfg(cfg: dict[str, Any], cfg_path: Path, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    import testset_utils

    gfp = cfg.get("gt_fixed_first_pose") or {}
    pose_id = int(gfp.get("pose_id") or 0)
    sample_meta = dict(meta or cfg.get("_render_meta") or {})
    return {
        "sample_id": testset_utils._safe_name(f"paper_{cfg['idx']}_{cfg['cue']}"),
        "testset": "iconic",
        "cue_idx": int(cfg["idx"]),
        "cue": str(cfg["cue"]),
        "config_path": str(cfg_path.resolve()),
        "gif_path": str(cfg_path),
        "selected_pose_id": pose_id,
        "meta": sample_meta,
    }


def _movement_params_from_cfg(cfg: dict[str, Any]) -> tuple[str, str]:
    for step in cfg.get("movements") or []:
        if step.get("type") != "movement":
            continue
        p = step.get("parameters") or {}
        mv = p.get("movement") or p
        joints = mv.get("joints") or []
        if joints:
            j0 = joints[0]
            joint = str(j0.get("joint") or "shoulder")
            axis = str(j0.get("axis") or "z")
            degs = j0.get("degrees") or {}
            if isinstance(degs, dict):
                for cand in ("x", "y", "z"):
                    if cand in degs:
                        axis = cand
                        break
            return joint, axis
        joint = str(p.get("joint") or mv.get("joint") or "shoulder")
        axis = "z"
        dirs = p.get("directions") or mv.get("directions") or []
        if dirs:
            degs = dirs[0].get("degrees") or {}
            for cand in ("x", "y", "z"):
                if cand in degs:
                    axis = cand
                    break
        return joint, axis
    return "shoulder", "z"


def _is_arm_sim_robot(robot: str) -> bool:
    r = (robot or "").strip()
    return r not in ("Tiago",) and not r.startswith("Go2") and "Spot" not in r and r not in ("Unitree", "UnitreeA1")


def pose_sim_png_for_cfg(
    cfg: dict[str, Any],
    robot: str,
    cfg_path: Path,
    out_png: Path,
    *,
    force: bool = False,
    meta: dict[str, Any] | None = None,
) -> Path | None:
    """Static first-pose frame from simulator (all embodiments)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file() and not force:
        return out_png
    pose_only = dict(cfg)
    movs = cfg.get("movements") or []
    if movs:
        pose_only["movements"] = [movs[0]]
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([pose_only], indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import testset_utils

        sample = _sample_for_cfg(pose_only, cfg_path, meta=meta)
        img, _ = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            robot,
            HZ,
            canonical="first_frame_trajectory",
            force=force,
        )
        img.save(out_png, format="PNG", optimize=False)
        return out_png
    except Exception as e:
        print(f"[pose sim fail] {cfg.get('cue')} ({robot}): {e}", flush=True)
        return None


def pose_sim_plain_png_for_cfg(
    cfg: dict[str, Any],
    robot: str,
    cfg_path: Path,
    out_png: Path,
    *,
    force: bool = False,
    meta: dict[str, Any] | None = None,
) -> Path | None:
    """Static simulator frame without trajectory overlays (plain RGB PNG)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file() and not force:
        return out_png
    pose_only = dict(cfg)
    movs = cfg.get("movements") or []
    if movs:
        pose_only["movements"] = [movs[0]]
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([pose_only], indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import testset_utils

        sample = _sample_for_cfg(pose_only, cfg_path, meta=meta)
        bundle = testset_utils.get_sim_bundle(sample, robot, HZ, force=force)
        frames = bundle.get("frames") or []
        if not frames:
            return None
        idx = min(len(frames) - 1, max(0, len(frames) // 2))
        Image.fromarray(frames[idx]).convert("RGB").save(out_png, format="PNG", optimize=False)
        return out_png
    except Exception as e:
        print(f"[pose plain sim fail] {cfg.get('cue')} ({robot}): {e}", flush=True)
        return None


def _movement_joint_screen_point(
    sample: dict[str, Any],
    cfg: dict[str, Any],
    *,
    robot: str = ROBOT,
    force: bool = False,
) -> tuple[int, int] | None:
    """Project the moving joint body to screen coords at mid-movement (manipulator only)."""
    if not _is_arm_sim_robot(robot):
        return None
    arm_pkg = _ROBOTARM
    vlm_pkg = _REPO / "adhoc" / "vlm_test"
    for p in (vlm_pkg, arm_pkg):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    arm_root = str(arm_pkg.resolve())
    try:
        sys.path.remove(arm_root)
    except ValueError:
        pass
    sys.path.insert(0, arm_root)

    import mujoco
    import testset_utils
    from legacy.vlm_pose_benchmark import _project_3d
    from motion_generation import MotionGenerator, _select_initial_poses

    joint_pref, axis = _movement_params_from_cfg(cfg)
    config_path = str(sample["config_path"])
    temp_output_root = (_REPO / "adhoc" / "test" / "_traj_tmp").resolve()
    temp_output_root.mkdir(parents=True, exist_ok=True)

    gen = None
    body_records: list[np.ndarray | None] = []
    try:
        jsonl_path = testset_utils.pose_database_jsonl()
        gen = MotionGenerator(
            robot_name=robot,
            jsonl_path=str(jsonl_path),
            output_dir=str(temp_output_root),
            capture_image_width=512,
            capture_image_height=512,
            hz=HZ,
        )
        pose_id = sample.get("selected_pose_id") or sample.get("pose_index")
        if pose_id is None:
            gfp = cfg.get("gt_fixed_first_pose")
            if gfp:
                matching = gen._find_matching_poses(gfp)
                selected = _select_initial_poses(matching, gfp, 1)
                if selected:
                    pose_id = selected[0]["pose_id"]

        orig_capture = gen._capture_image

        def _capture_with_joint():
            try:
                _, jname, _, _, _ = gen._select_joint(axis, joint_pref)
                model = gen.env.sim.model._model
                data = gen.env.sim.data._data
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                bid = int(model.jnt_bodyid[jid])
                body_records.append(np.array(data.xpos[bid], dtype=float).copy())
            except Exception:
                body_records.append(None)
            return orig_capture()

        gen._capture_image = _capture_with_joint
        try:
            gen._set_joint_positions(gen.initial_joint_pos)
            gen.execute_cue(
                cue=sample["cue"],
                cue_idx=int(sample["cue_idx"]),
                pose_index=pose_id,
                config_path=config_path,
                hz=HZ,
                save_gif=False,
                overlay_progress_bar=False,
            )
        finally:
            gen._capture_image = orig_capture

        valid = [p for p in body_records if p is not None]
        if not valid:
            return None
        mid_body = valid[len(valid) // 2]

        cam_id = gen.env.sim.model.camera_name2id("frontview")
        cam_pos = gen.env.sim.data.cam_xpos[cam_id].copy()
        cam_rot = gen.env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = gen.env.sim.model.cam_fovy[cam_id]
        pt = _project_3d(mid_body, cam_pos, cam_rot, fovy, 512)
        return pt
    except Exception as e:
        print(f"[movement joint fail] {cfg.get('cue')}: {e}", flush=True)
        return None
    finally:
        if gen is not None:
            try:
                gen.env.close()
            except Exception:
                pass


def _draw_black_path_line(img: Image.Image, meta: dict[str, Any]) -> None:
    pts = meta.get("trajectory_screen_xy") or []
    if len(pts) < 2:
        return
    p0 = (int(pts[0]["x"]), int(pts[0]["y"]))
    p1 = (int(pts[-1]["x"]), int(pts[-1]["y"]))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if (dx * dx + dy * dy) ** 0.5 < 48:
        return
    draw = ImageDraw.Draw(img)
    draw.line([p0, p1], fill=(0, 0, 0), width=4)


def _draw_black_joint_circle(img: Image.Image, center: tuple[int, int], *, radius: int = 10) -> None:
    x, y = center
    draw = ImageDraw.Draw(img)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(0, 0, 0))


def _enrich_path_meta(meta: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    pts = meta.get("trajectory_screen_xy") or []
    if len(pts) >= 2:
        return meta
    traj = bundle.get("trajectory") or []
    if len(traj) < 2:
        return meta
    import testset_utils

    cam_pos = bundle["cam_pos"]
    cam_rot = bundle["cam_rot"]
    fovy = float(bundle["fovy"])
    width = int(bundle.get("width") or 512)
    screen = testset_utils._project_ee_to_screen(traj, cam_pos, cam_rot, fovy, width)
    if len(screen) >= 2:
        meta = dict(meta)
        meta["trajectory_screen_xy"] = [{"x": int(x), "y": int(y)} for x, y in screen]
    return meta


def alpha_trajectory_for_cfg(
    cfg: dict[str, Any],
    cfg_path: Path,
    out_png: Path,
    *,
    robot: str = ROBOT,
    force: bool = False,
    stack_count: int = 10,
    blend_weight: float = 0.64,
    crop_top_frac: float = 0.0,
    annotate: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path | None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file() and not force and annotate is None:
        return out_png
    try:
        import testset_utils

        sample = _sample_for_cfg(cfg, cfg_path, meta=meta)
        img, meta = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            robot,
            HZ,
            canonical="alpha_frame_trajectory",
            force=force,
            stack_count=stack_count,
            blend_weight=blend_weight,
        )
        if annotate == "path" and robot != "Tiago":
            bundle = testset_utils.get_sim_bundle(sample, robot, HZ, force=force)
            meta = _enrich_path_meta(meta, bundle)
            _draw_black_path_line(img, meta)
        elif annotate == "movement":
            joint_pt = _movement_joint_screen_point(sample, cfg, robot=robot, force=force)
            if joint_pt:
                _draw_black_joint_circle(img, joint_pt)
        if crop_top_frac > 0:
            w, h = img.size
            cut = max(1, int(h * crop_top_frac))
            img = img.crop((0, cut, w, h))
        img.save(out_png, format="PNG", optimize=False)
        return out_png
    except Exception as e:
        print(f"[alpha fail] {cfg.get('cue')}: {e}", flush=True)
        return None


def alpha_trajectory_for_row(
    row: dict[str, Any],
    out_png: Path,
    *,
    force: bool = False,
    stack_count: int = 10,
    blend_weight: float = 0.64,
    crop_top_frac: float = 0.0,
) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file() and not force:
        return out_png
    gif = pick_gif(str(row["cue"]))
    if not gif:
        return None
    try:
        import testset_utils

        pose = _first_pose(row)
        pose_id = int(pose.get("pose_id") or 0)
        sample = {
            "sample_id": testset_utils._safe_name(f"paper_{row['idx']}_{row['cue']}"),
            "testset": "iconic",
            "cue_idx": int(row["idx"]),
            "cue": row["cue"],
            "config_path": str(POSE_CFG),
            "gif_path": str(gif),
            "selected_pose_id": pose_id,
            "meta": {},
        }
        img, _ = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            ROBOT,
            HZ,
            canonical="alpha_frame_trajectory",
            force=force,
            stack_count=stack_count,
            blend_weight=blend_weight,
        )
        if crop_top_frac > 0:
            w, h = img.size
            cut = max(1, int(h * crop_top_frac))
            img = img.crop((0, cut, w, h))
        img.save(out_png, format="PNG", optimize=False)
        return out_png
    except Exception as e:
        print(f"[alpha fail] {row.get('cue')}: {e}", flush=True)
        return None


def mid_frame_from_gif(gif: Path, out_png: Path) -> Path | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.is_file():
        return out_png
    try:
        im = Image.open(gif)
        n = getattr(im, "n_frames", 1)
        im.seek(n // 2)
        im.convert("RGB").save(out_png)
        return out_png
    except Exception:
        return None


def _font(size: int = 10) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_box(
    img: Image.Image,
    text: str,
    *,
    color: tuple[int, int, int, int],
    corner: str = "top_right",
) -> Image.Image:
    """Overlay semi-transparent True/False label."""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(11)
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = base.size
    if corner == "top_right":
        x0, y0 = w - tw - 2 * pad - 4, 4
    else:
        x0, y0 = 4, 4
    draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad], fill=color)
    draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def mp4_last_frame_png(mp4: Path, out_png: Path) -> Path | None:
    """Extract last frame from pairwise MP4 (full alpha trajectory)."""
    ff = shutil.which("ffmpeg")
    if not ff or not mp4.is_file():
        return None
    out_png.parent.mkdir(parents=True, exist_ok=True)
    last_n = 8
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        pr = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(mp4),
            ],
            capture_output=True,
            text=True,
        )
        if pr.returncode == 0 and pr.stdout.strip().isdigit():
            last_n = max(0, int(pr.stdout.strip()) - 1)
    r = subprocess.run(
        [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"select=eq(n\\,{last_n})",
            "-vframes",
            "1",
            "-update",
            "1",
            str(out_png),
        ],
        capture_output=True,
    )
    return out_png if r.returncode == 0 and out_png.is_file() and out_png.stat().st_size > 0 else None


def gif_to_mp4(gif: Path, mp4: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    mp4.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gif), "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True,
    )
    return r.returncode == 0 and mp4.is_file()
