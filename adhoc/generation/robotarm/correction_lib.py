"""Instance-level correction: render → wrapped exp1/exp7 prompt (Correction Q1/Q2) → parse → re-render."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_GOOGLE = _REPO / "adhoc/generation/tiago"
import sys

for p in (_REPO, _HERE, _GOOGLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_prompts import (  # noqa: E402
    ExpId,
    InnerExpId,
    build_wrapped_correction_prompt,
    correction_indicates_verified_ok,
    parse_correction_response,
    row_from_parsed_config,
)
from render_paths import (  # noqa: E402
    RenderRole,
    cumulative_tile_path,
    render_exp_dir,
    render_gif_path,
    render_media_paths,
    manipulator_cue_render_dir,
    MANIPULATOR_RENDER_HZ,
    render_role_for_round,
    uses_new_renderer,
)
from pilot90_paths import (  # noqa: E402
    POSE_JSONL,
    load_config_list,
    load_gt_by_cue as load_manip_gt_by_cue,
    result_config_path as manip_result_config_path,
    upsert_config_row,
)

Domain = Literal["manipulator", "tiago"]


@dataclass
class RenderArtifacts:
    gif: Path | None = None
    mp4: Path | None = None
    pose_png: Path | None = None
    pose_pool_gif: Path | None = None
    cumulative_tile_gif: Path | None = None


@dataclass
class CorrectionResult:
    cue: str
    domain: Domain
    exp_id: ExpId
    model_tag: str
    round: int
    parent_round: int
    config_json: Path
    input_row: dict[str, Any]
    correction: dict[str, Any]
    is_correct: bool | None
    action: str
    output_row: dict[str, Any] | None = None
    media: RenderArtifacts = field(default_factory=RenderArtifacts)
    correction_json: Path | None = None
    config_snapshot_json: Path | None = None


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def correction_stem(round_n: int) -> str:
    """Round 1 → correction, round 2 → correction_correction, …"""
    if round_n < 1:
        raise ValueError(f"round must be >= 1, got {round_n}")
    return "_".join(["correction"] * round_n)


def round_from_stem(stem: str) -> int | None:
    if stem == "correction":
        return 1
    parts = stem.split("_")
    if parts and all(p == "correction" for p in parts):
        return len(parts)
    return None


def correction_cue_dir(
    domain: Domain,
    exp_id: ExpId,
    model_tag: str,
    cue: str,
) -> Path:
    return (
        _REPO
        / "data/results/verify"
        / domain
        / "correction"
        / f"exp{exp_id}"
        / model_tag
        / _safe_cue(cue)
    )


def config_snapshot_dir(
    domain: Domain,
    exp_id: ExpId,
    model_tag: str,
    cue: str,
) -> Path:
    return (
        _REPO
        / "data/results/motion_configs"
        / domain
        / "exp"
        / "correction"
        / model_tag
        / f"exp{exp_id}"
        / _safe_cue(cue)
    )


def correction_record_path(cue_dir: Path, round_n: int) -> Path:
    return cue_dir / f"{correction_stem(round_n)}.json"


def config_snapshot_path(
    domain: Domain,
    exp_id: ExpId,
    model_tag: str,
    cue: str,
    round_n: int,
) -> Path:
    return config_snapshot_dir(domain, exp_id, model_tag, cue) / f"{correction_stem(round_n)}_config.json"


def list_correction_rounds(cue_dir: Path) -> list[int]:
    if not cue_dir.is_dir():
        return []
    rounds: list[int] = []
    for p in cue_dir.glob("correction*.json"):
        if p.name.endswith("_config.json"):
            continue
        r = round_from_stem(p.stem)
        if r is not None:
            rounds.append(r)
    return sorted(set(rounds))


def next_correction_round(cue_dir: Path) -> int:
    existing = list_correction_rounds(cue_dir)
    return (max(existing) if existing else 0) + 1


def load_correction_record(cue_dir: Path, round_n: int) -> dict[str, Any]:
    path = correction_record_path(cue_dir, round_n)
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_correction_record(
    cue_dir: Path,
) -> tuple[dict[str, Any], int] | None:
    rounds = list_correction_rounds(cue_dir)
    if not rounds:
        return None
    r = max(rounds)
    return load_correction_record(cue_dir, r), r


def output_row_from_correction_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("output_row"):
        return record["output_row"]
    if record.get("corrected_row"):
        return record["corrected_row"]
    return record["original_row"]


def write_config_snapshot(path: Path, row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_row_for_correction(
    *,
    domain: Domain,
    exp_id: ExpId,
    cue: str,
    model_tag: str,
    base_config_path: Path,
    chain_from_latest: bool = True,
) -> tuple[dict[str, Any], int, Path]:
    """Return (input_row, parent_round, config_path_for_render).

    parent_round=0 → input came from base generation config.
    parent_round=N → input came from round N correction output.
    """
    cue_dir = correction_cue_dir(domain, exp_id, model_tag, cue)
    if chain_from_latest:
        latest = load_latest_correction_record(cue_dir)
        if latest is not None:
            record, parent_round = latest
            row = output_row_from_correction_record(record)
            snap = config_snapshot_path(domain, exp_id, model_tag, cue, parent_round)
            if not snap.is_file():
                write_config_snapshot(snap, row)
            return row, parent_round, snap

    row = load_config_row(base_config_path, cue=cue)
    return row, 0, base_config_path


def _save_gif_frames(frames: list, gif_path: Path, *, duration_ms: int = 50) -> None:
    if str(_REPO / "adhoc/generation") not in sys.path:
        sys.path.insert(0, str(_REPO / "adhoc/generation"))
    from gif_encode import save_gif_frames

    save_gif_frames(frames, gif_path, duration_ms=duration_ms)


def _gif_first_frame_png(gif_path: Path, png_path: Path) -> Path:
    from PIL import Image

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(gif_path) as im:
        im.seek(0)
        im.convert("RGB").save(png_path)
    return png_path


def _ensure_mp4(gif_path: Path, mp4_path: Path) -> Path | None:
    if not gif_path.is_file():
        return None
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.is_file() and mp4_path.stat().st_mtime >= gif_path.stat().st_mtime:
        return mp4_path
    from shutil import which

    if not which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(gif_path),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return mp4_path if mp4_path.is_file() else None


def default_config_path(domain: Domain, model_tag: str, exp_id: ExpId = "1") -> Path:
    if domain == "manipulator":
        return manip_result_config_path(exp_id, model_tag)
    from pilot40_paths import result_config_path  # noqa: WPS433

    return result_config_path(exp_id, model_tag)


def default_render_dir(
    domain: Domain,
    model_tag: str,
    cue: str,
    exp_id: ExpId = "1",
    round_n: int = 1,
) -> Path:
    del cue, round_n  # flat exp directory; filenames encode cue/round
    return render_exp_dir(domain, model_tag, exp_id)


def load_config_row(config_path: Path, *, cue: str | None = None, idx: int | None = None) -> dict[str, Any]:
    rows = load_config_list(config_path)
    if cue is not None:
        for row in rows:
            if row.get("cue") == cue:
                return row
        raise KeyError(f"cue not found in {config_path}: {cue!r}")
    if idx is not None:
        for row in rows:
            if int(row.get("idx", -1)) == int(idx):
                return row
        raise KeyError(f"idx not found in {config_path}: {idx!r}")
    raise ValueError("Provide cue or idx")


def cue_description(domain: Domain, row: dict[str, Any]) -> str:
    desc = str(row.get("description") or "").strip()
    if desc:
        return desc
    cue = str(row.get("cue", ""))
    if domain == "manipulator":
        gt = load_manip_gt_by_cue().get(cue) or {}
        return str(gt.get("description") or "")
    from pilot40_paths import load_gt_by_cue  # noqa: WPS433

    gt = load_gt_by_cue().get(cue) or {}
    return str(gt.get("description") or "")


def _pose_index_from_row(row: dict[str, Any]) -> int | None:
    gfp = row.get("gt_fixed_first_pose")
    if isinstance(gfp, dict) and gfp.get("pose_id") is not None:
        return int(gfp["pose_id"])
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose")
        if isinstance(pose, dict) and pose.get("pose_id") is not None:
            return int(pose["pose_id"])
        break
    return None


def motion_gif_path_for_round(
    domain: Domain,
    model_tag: str,
    exp_id: ExpId,
    cue: str,
    idx: int,
    attempt_round: int,
) -> Path:
    """Top-1 motion GIF path: round 0 = generation, round N = correction round N."""
    role = render_role_for_round(attempt_round)
    return render_gif_path(domain, model_tag, exp_id, idx, cue, role)


def load_cumulative_prior_attempts(
    *,
    domain: Domain,
    exp_id: ExpId,
    model_tag: str,
    cue: str,
    base_config_path: Path,
    parent_round: int,
) -> list[tuple[str, dict[str, Any], int]]:
    """Rows for cumulative correction: original + correction 1..parent_round."""
    if parent_round < 1:
        return []
    cue_dir = correction_cue_dir(domain, exp_id, model_tag, cue)
    base_row = load_config_row(base_config_path, cue=cue)
    out: list[tuple[str, dict[str, Any], int]] = [("Original (exp1)", base_row, 0)]
    for rn in range(1, parent_round + 1):
        rec = load_correction_record(cue_dir, rn)
        row = output_row_from_correction_record(rec)
        out.append((f"Correction {rn}", row, rn))
    return out


_COMPOSE_BEFORE_RENDER_ENV = "ROBOSUITE_COMPOSE_BEFORE_RENDER"
_GOOGLE_RENDER_WARMED_ENV = "ROBOSUITE_TIAGO_RENDER_WARMED"


def _mark_compose_before_render() -> None:
    os.environ[_COMPOSE_BEFORE_RENDER_ENV] = "1"


def _consume_compose_before_render() -> bool:
    return os.environ.pop(_COMPOSE_BEFORE_RENDER_ENV, None) == "1"


def _mark_google_render_warmed() -> None:
    os.environ[_GOOGLE_RENDER_WARMED_ENV] = "1"


def _google_render_needs_prime(*, after_compose_prime: bool) -> bool:
    if after_compose_prime:
        return False
    if os.environ.get(_GOOGLE_RENDER_WARMED_ENV) != "1":
        return True
    return os.environ.get(_COMPOSE_BEFORE_RENDER_ENV) == "1"


def compose_tiled_motion_gif(gif_paths: list[Path], out_path: Path) -> Path:
    """Horizontally tile motion GIFs (sync-play) into one GIF for VLM input."""
    from PIL import Image

    if not gif_paths:
        raise ValueError("compose_tiled_motion_gif: no input GIFs")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_frames: list[list[Image.Image]] = []
    for p in gif_paths:
        if not p.is_file():
            raise FileNotFoundError(f"missing motion GIF for cumulative tile: {p}")
        with Image.open(p) as im:
            frames = []
            try:
                while True:
                    frames.append(im.copy().convert("RGB"))
                    im.seek(im.tell() + 1)
            except EOFError:
                pass
        if not frames:
            raise ValueError(f"empty GIF: {p}")
        cell_frames.append(frames)
    max_len = max(len(fs) for fs in cell_frames)
    grid_frames: list[Image.Image] = []
    for t in range(max_len):
        thumbs = [fs[min(t, len(fs) - 1)] for fs in cell_frames]
        w, h = thumbs[0].size
        canvas = Image.new("RGB", (w * len(thumbs), h))
        for i, im in enumerate(thumbs):
            canvas.paste(im, (i * w, 0))
        grid_frames.append(canvas)
    _save_gif_frames(grid_frames, out_path)
    _mark_compose_before_render()
    return out_path


def build_cumulative_vlm_media(
    domain: Domain,
    exp_id: ExpId,
    model_tag: str,
    cue: str,
    *,
    parent_round: int,
    base_config_path: Path,
    out_dir: Path,
) -> tuple[RenderArtifacts, list[tuple[str, dict[str, Any]]]]:
    """Tiled motion GIF + labeled prior configs for cumulative correction."""
    attempts = load_cumulative_prior_attempts(
        domain=domain,
        exp_id=exp_id,
        model_tag=model_tag,
        cue=cue,
        base_config_path=base_config_path,
        parent_round=parent_round,
    )
    if len(attempts) < 2:
        raise ValueError("cumulative media requires parent_round >= 1")
    idx = int(attempts[0][1].get("idx", -1))
    gif_paths = [
        motion_gif_path_for_round(domain, model_tag, exp_id, cue, idx, rnd)
        for _, _, rnd in attempts
    ]
    tiled_path = cumulative_tile_path(domain, model_tag, exp_id, idx, cue, len(gif_paths))
    compose_tiled_motion_gif(gif_paths, tiled_path)
    prior_for_prompt = [(label, row) for label, row, _ in attempts]
    media = RenderArtifacts(
        gif=tiled_path,
        cumulative_tile_gif=tiled_path,
        pose_png=_gif_first_frame_png(tiled_path, tiled_path.parent / f"{tiled_path.stem}_pose.png"),
    )
    return media, prior_for_prompt


def _latest_manipulator_gif(variant_dir: Path) -> Path | None:
    iwa = variant_dir / "IIWA"
    if not iwa.is_dir():
        return None
    cands = sorted(iwa.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_manipulator_new_render_media(
    model_tag: str,
    exp_id: ExpId,
    row: dict[str, Any],
) -> RenderArtifacts:
    cue = str(row.get("cue", ""))
    nr = manipulator_cue_render_dir(model_tag, exp_id, cue, "new_render")
    gif = _latest_manipulator_gif(nr)
    if gif is None:
        return RenderArtifacts()
    mp4 = _ensure_mp4(gif, gif.with_suffix(".mp4"))
    png = _gif_first_frame_png(gif, nr / f"{cue}_pose.png")
    return RenderArtifacts(gif=gif, mp4=mp4, pose_png=png)


def _load_exp_media(
    domain: Domain,
    model_tag: str,
    exp_id: ExpId,
    row: dict[str, Any],
) -> RenderArtifacts:
    """Reuse rendered generation GIF/MP4."""
    if domain == "manipulator":
        return _load_manipulator_new_render_media(model_tag, exp_id, row)
    if domain != "tiago":
        return RenderArtifacts()
    idx = int(row.get("idx", -1))
    cue = str(row.get("cue", ""))
    gif, mp4, png = render_media_paths(domain, model_tag, exp_id, idx, cue, "gen")
    if not gif.is_file():
        return RenderArtifacts()
    if not png.is_file():
        png = _gif_first_frame_png(gif, png)
    return RenderArtifacts(gif=gif, mp4=mp4 if mp4.is_file() else None, pose_png=png)


def _import_tiago_render():
    """Import tiago legacy renderer (shadowed by robotarm ``legacy`` on sys.path).

    Keep a single module instance — reloading breaks MuJoCo/IKSolver globals and
    causes flickery GIFs on the 2nd+ cue in a batch.
    """
    import importlib.util

    mod_path = _GOOGLE / "legacy" / "render_mobile_config.py"
    mod_name = "tiago_render_mobile_config"
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return mod._make_env, mod.render_config, mod


def _import_pose_bank_topk_tiles():
    """Import pose_bank_topk_tiles without robotarm ``legacy`` shadowing."""
    import importlib.util

    mod_path = _GOOGLE / "pose_bank_topk_tiles.py"
    spec = importlib.util.spec_from_file_location("tiago_pose_bank_topk_tiles", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    google_path = str(_GOOGLE)
    sys.path.insert(0, google_path)
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == google_path:
            sys.path.pop(0)
    return mod


def render_instance(
    domain: Domain,
    row: dict[str, Any],
    *,
    config_path: Path,
    render_dir: Path | None = None,
    model_tag: str = "gemini-2.5-pro",
    exp_id: ExpId = "1",
    round_n: int = 1,
    render_role: RenderRole | None = None,
    ignore_xyz: bool | None = None,
    topk: int | None = None,
    manipulator_variant: Literal["new_render", "correction"] | None = None,
    _after_compose_prime: bool = False,
) -> RenderArtifacts:
    cue = str(row.get("cue", ""))
    idx = int(row.get("idx", -1))
    role = render_role or render_role_for_round(round_n)
    out = render_dir or render_exp_dir(domain, model_tag, exp_id)
    out.mkdir(parents=True, exist_ok=True)
    if ignore_xyz is None:
        ignore_xyz = not uses_new_renderer(domain, exp_id)

    if domain == "manipulator":
        robotarm = Path(__file__).resolve().parent
        if sys.path[0] != str(robotarm):
            sys.path.insert(0, str(robotarm))
        from legacy.motion_generation_core import generate  # noqa: WPS433

        variant = manipulator_variant or (
            "correction" if role == "corr" else "new_render"
        )
        out = manipulator_cue_render_dir(model_tag, exp_id, cue, variant)
        out.mkdir(parents=True, exist_ok=True)
        robot_out = out / "IIWA"
        robot_out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
        before = set(robot_out.glob("*.gif"))
        generate(
            robot="IIWA",
            cue=cue,
            cue_idx=idx,
            pose_index=None,
            jsonl_path=str(POSE_JSONL),
            config_path=str(config_path),
            output_dir=str(out),
            hz=MANIPULATOR_RENDER_HZ,
            top_k=1,
            ignore_xyz=False,
        )
        after = set(robot_out.glob("*.gif"))
        new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        gif = new_files[0] if new_files else _latest_manipulator_gif(out)
        if gif is None:
            return RenderArtifacts()
        png = _gif_first_frame_png(gif, out / f"{cue}_pose.png")
        mp4 = _ensure_mp4(gif, out / f"{cue}.mp4")
        return RenderArtifacts(gif=gif, mp4=mp4, pose_png=png)

    gif_path, mp4_path, pose_png_path = render_media_paths(
        domain, model_tag, exp_id, idx, cue, role
    )
    pool_path = gif_path.parent / f"{gif_path.stem}_pose_pool.gif"
    if _google_render_needs_prime(after_compose_prime=_after_compose_prime):
        _consume_compose_before_render()
        tmp = out.parent / f".compose_prime_{_safe_cue(cue)}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        render_instance(
            domain,
            row,
            config_path=config_path,
            render_dir=tmp,
            model_tag=model_tag,
            exp_id=exp_id,
            round_n=round_n,
            render_role=role,
            ignore_xyz=ignore_xyz,
            topk=topk,
            _after_compose_prime=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)
        _mark_google_render_warmed()
    _make_env, render_config, _render_mod = _import_tiago_render()
    env = _make_env()
    try:
        frames = render_config(
            row,
            env=env,
            overlay_progress_bar=True,
            progress_bar_style="typed",
            ignore_xyz=ignore_xyz,
        )
        if frames:
            _save_gif_frames(frames, gif_path)
        pose_pool_gif: Path | None = None
        if ignore_xyz:
            pbt = _import_pose_bank_topk_tiles()
            pose = pbt.first_pose_from_row(row)
            if pose:
                pbt.render_plausible_pool_tile(
                    env,
                    pose,
                    pool_path,
                    ignore_xyz=True,
                    topk=topk,
                    row=row,
                    overlay_progress_bar=False,
                    main_gif_path=gif_path,
                )
                if pool_path.is_file():
                    pose_pool_gif = pool_path
    finally:
        try:
            _render_mod._TIAGO_ARM_IK_CACHE.pop(id(env), None)
            env.close()
        except Exception:
            pass
    if not gif_path.is_file():
        return RenderArtifacts()
    png = _gif_first_frame_png(gif_path, pose_png_path)
    mp4 = _ensure_mp4(gif_path, mp4_path)
    return RenderArtifacts(gif=gif_path, mp4=mp4, pose_png=png, pose_pool_gif=pose_pool_gif)


def _prepare_correction_media(
    domain: Domain,
    exp_id: ExpId,
    media: RenderArtifacts,
) -> tuple[list[Any] | None, list[str] | None, dict[str, str]]:
    """Return (pil_images, video_paths, meta) for VLM input.

    exp1 (manipulator + tiago) → full motion MP4 or animated GIF.
    exp7 → MP4 video.
    """
    meta: dict[str, str] = {}
    if exp_id == "7":
        if media.mp4 and media.mp4.is_file():
            meta = {"kind": "video/mp4", "path": str(media.mp4)}
            return None, [str(media.mp4.resolve())], meta
        if media.gif and media.gif.is_file():
            meta = {"kind": "image/gif", "path": str(media.gif)}
            return None, None, meta  # gemini raw gif handled separately
        raise FileNotFoundError("Rendered motion video/gif missing — run render first")

    if media.mp4 and media.mp4.is_file():
        meta = {"kind": "video/mp4", "path": str(media.mp4)}
        return None, [str(media.mp4.resolve())], meta
    if media.gif and media.gif.is_file():
        meta = {"kind": "image/gif", "path": str(media.gif)}
        return None, None, meta
    raise FileNotFoundError("Rendered motion GIF/MP4 missing — run render first")


def run_correction_vlm(
    domain: Domain,
    exp_id: ExpId,
    row: dict[str, Any],
    media: RenderArtifacts,
    *,
    model: str = "gemini-2.5-pro",
    backend: str = "gemini",
    vlm: Any | None = None,
    inner_exp_id: InnerExpId | None = None,
    prior_attempts: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Single VLM call: inner exp prompt wrapped with Correction Q1/Q2."""
    cue = str(row.get("cue", ""))
    description = cue_description(domain, row)
    prompt = build_wrapped_correction_prompt(
        domain,
        exp_id,
        cue=cue,
        row=row,
        description=description,
        backend=backend,
        inner_exp_id=inner_exp_id,
        prior_attempts=prior_attempts,
    )

    images, videos, media_meta = _prepare_correction_media(domain, exp_id, media)
    raw_gif_path: Path | None = None
    if media_meta.get("kind") == "image/gif":
        raw_gif_path = Path(media_meta["path"])

    raw_text = ""
    if vlm is not None:
        if raw_gif_path is not None:
            raise NotImplementedError("VLMClient path does not yet support raw GIF; use backend=gemini")
        raw_text = vlm.generate(prompt, images=images, videos=videos)
    elif backend.lower() == "gemini":
        from google import genai
        from google.genai import types

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        client = genai.Client(api_key=key)
        parts: list[Any] = []
        if raw_gif_path is not None and raw_gif_path.is_file():
            parts.append(
                types.Part.from_bytes(data=raw_gif_path.read_bytes(), mime_type="image/gif")
            )
        for img in images or []:
            import io

            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
        for vpath in videos or []:
            parts.append(types.Part.from_bytes(data=Path(vpath).read_bytes(), mime_type="video/mp4"))
        parts.append(prompt)
        resp = client.models.generate_content(model=model, contents=parts)
        raw_text = (resp.text or "").strip()
    else:
        from vlm_client import VLMClient

        vlm = vlm or VLMClient(backend=backend, model=model)
        raw_text = vlm.generate(prompt, images=images, videos=videos)

    parsed = parse_correction_response(raw_text, exp_id=exp_id)
    parsed["prompt_exp_id"] = inner_exp_id if inner_exp_id is not None else exp_id
    parsed["pipeline_exp_id"] = exp_id
    parsed["vlm_media"] = media_meta
    return parsed


def save_correction_record(
    result: CorrectionResult,
    *,
    out_dir: Path | None = None,
) -> Path:
    cue = result.cue
    cue_dir = out_dir or correction_cue_dir(
        result.domain, result.exp_id, result.model_tag, cue
    )
    cue_dir.mkdir(parents=True, exist_ok=True)
    path = correction_record_path(cue_dir, result.round)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "domain": result.domain,
        "exp_id": result.exp_id,
        "model_tag": result.model_tag,
        "round": result.round,
        "parent_round": result.parent_round,
        "correction_stem": correction_stem(result.round),
        "cue": cue,
        "base_config_json": str(result.config_json),
        "config_snapshot_json": (
            str(result.config_snapshot_json) if result.config_snapshot_json else None
        ),
        "is_correct": result.is_correct,
        "action": result.action,
        "correction": result.correction,
        "input_row": result.input_row,
        "output_row": result.output_row,
        "media": {
            "gif": str(result.media.gif) if result.media.gif else None,
            "mp4": str(result.media.mp4) if result.media.mp4 else None,
            "pose_png": str(result.media.pose_png) if result.media.pose_png else None,
            "cumulative_tile_gif": (
                str(result.media.cumulative_tile_gif) if result.media.cumulative_tile_gif else None
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result.correction_json = path
    return path


def run_instance_correction(
    *,
    domain: Domain,
    exp_id: ExpId,
    cue: str,
    config_path: Path | None = None,
    model_tag: str = "gemini-2.5-pro",
    model: str | None = None,
    backend: str = "gemini",
    vlm: Any | None = None,
    render_dir: Path | None = None,
    verify_only: bool = False,
    skip_render: bool = False,
    save_config: bool = True,
    update_main_config: bool = False,
    chain_from_latest: bool = True,
    rerender: bool = True,
    inner_exp_id: InnerExpId | None = None,
    ignore_xyz: bool = False,
    topk: int | None = None,
) -> CorrectionResult:
    cfg_path = config_path or default_config_path(domain, model_tag, exp_id)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    model_name = model or model_tag
    cue_dir = correction_cue_dir(domain, exp_id, model_tag, cue)
    round_n = next_correction_round(cue_dir)

    input_row, parent_round, render_cfg_path = load_row_for_correction(
        domain=domain,
        exp_id=exp_id,
        cue=cue,
        model_tag=model_tag,
        base_config_path=cfg_path,
        chain_from_latest=chain_from_latest,
    )

    out_render_dir = render_dir or default_render_dir(
        domain, model_tag, cue, exp_id, round_n
    )
    prior_attempts: list[tuple[str, dict[str, Any]]] | None = None
    media = RenderArtifacts()
    use_cumulative = (
        domain == "tiago"
        and exp_id == "1"
        and parent_round >= 1
        and not skip_render
    )
    if skip_render:
        media = _load_exp_media(domain, model_tag, exp_id, input_row)
    elif use_cumulative:
        media, prior_attempts = build_cumulative_vlm_media(
            domain,
            exp_id,
            model_tag,
            cue,
            parent_round=parent_round,
            base_config_path=cfg_path,
            out_dir=out_render_dir,
        )
    else:
        if domain == "manipulator":
            media = _load_manipulator_new_render_media(model_tag, exp_id, input_row)
            if not media.gif:
                media = render_instance(
                    domain,
                    input_row,
                    config_path=render_cfg_path,
                    model_tag=model_tag,
                    exp_id=exp_id,
                    round_n=round_n,
                    manipulator_variant="new_render",
                    ignore_xyz=ignore_xyz,
                    topk=topk,
                )
        else:
            media = render_instance(
                domain,
                input_row,
                config_path=render_cfg_path,
                render_dir=out_render_dir,
                model_tag=model_tag,
                exp_id=exp_id,
                round_n=round_n,
                ignore_xyz=ignore_xyz,
                topk=topk,
            )
            if domain == "tiago" and not media.gif:
                media = _load_exp_media(domain, model_tag, exp_id, input_row)

    correction = run_correction_vlm(
        domain,
        exp_id,
        input_row,
        media,
        model=model_name,
        backend=backend,
        vlm=vlm,
        inner_exp_id=inner_exp_id,
        prior_attempts=prior_attempts,
    )
    is_correct = correction.get("is_correct")
    if correction_indicates_verified_ok(correction):
        is_correct = True
        correction["is_correct"] = True

    action = "none"
    output_row: dict[str, Any] | None = None
    config_snapshot_json: Path | None = None

    if is_correct is True:
        action = "verified_ok"
        output_row = input_row
        if domain == "tiago" and exp_id == "1" and save_config:
            config_snapshot_json = config_snapshot_path(domain, exp_id, model_tag, cue, round_n)
            write_config_snapshot(config_snapshot_json, output_row)
        if domain == "tiago" and exp_id == "1" and parent_round >= 1:
            idx = int(input_row.get("idx", -1))
            parent_gif = motion_gif_path_for_round(
                domain, model_tag, exp_id, cue, idx, parent_round
            )
            _, out_gif, out_pose = render_media_paths(
                domain, model_tag, exp_id, idx, cue, render_role_for_round(round_n)
            )
            if parent_gif.is_file():
                out_gif.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parent_gif, out_gif)
                media.gif = out_gif
                media.mp4 = _ensure_mp4(out_gif, out_gif.with_suffix(".mp4"))
                media.pose_png = _gif_first_frame_png(parent_gif, out_pose)
    elif verify_only:
        action = "verify_only"
        output_row = input_row
    else:
        config_parsed = correction.get("config_parsed")
        if not isinstance(config_parsed, dict) or not config_parsed.get("movements"):
            raise ValueError(
                f"Correction Q1=no but no valid config JSON in response "
                f"(parse_error={correction.get('parse_error')!r})"
            )
        output_row = row_from_parsed_config(
            domain,
            exp_id,
            input_row,
            config_parsed,
            model=model_name,
        )
        output_row["correction_q1"] = correction.get("correction_q1")
        output_row["correction_q2"] = correction.get("correction_q2")
        if correction.get("inner_reasoning"):
            output_row["reasoning"] = correction["inner_reasoning"]
        action = "corrected"
        config_snapshot_json = config_snapshot_path(domain, exp_id, model_tag, cue, round_n)
        if save_config:
            write_config_snapshot(config_snapshot_json, output_row)
        if update_main_config:
            upsert_config_row(cfg_path, output_row)

    if output_row is not None and action == "corrected" and rerender and not verify_only:
        rerender_cfg = config_snapshot_json
        if rerender_cfg is None or not rerender_cfg.is_file():
            rerender_cfg = config_snapshot_path(domain, exp_id, model_tag, cue, round_n)
            write_config_snapshot(rerender_cfg, output_row)
            config_snapshot_json = rerender_cfg
        media = render_instance(
            domain,
            output_row,
            config_path=rerender_cfg,
            render_dir=render_dir or out_render_dir,
            model_tag=model_tag,
            exp_id=exp_id,
            round_n=round_n,
            manipulator_variant="correction" if domain == "manipulator" else None,
            ignore_xyz=ignore_xyz,
            topk=topk,
        )

    result = CorrectionResult(
        cue=cue,
        domain=domain,
        exp_id=exp_id,
        model_tag=model_tag,
        round=round_n,
        parent_round=parent_round,
        config_json=cfg_path,
        input_row=input_row,
        correction=correction,
        is_correct=is_correct if isinstance(is_correct, bool) else None,
        action=action,
        output_row=output_row,
        media=media,
        config_snapshot_json=config_snapshot_json,
    )
    save_correction_record(result)
    return result
