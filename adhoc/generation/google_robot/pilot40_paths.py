"""Canonical paths for Google Robot pilot-90 (90 cues) — mirrors manipulator pilot90_paths."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
ROBOT = "google_robot"

PROMPT_EXP_DIR = _REPO / "data/seed/prompt" / ROBOT / "exp"
PROMPT_LEGACY_DIR = _REPO / "data/seed/prompt" / ROBOT
SHOTS = _REPO / "data/seed/shots" / ROBOT / "shot_configs_pilot40_mobile.json"
FEWSHOT_SHOTS = _REPO / "data/seed/shots" / ROBOT / "diverse_shots_mobile.json"
CUES_YAML = _REPO / "data/seed/yml/_cues_new.yml"
MANIFEST_TSV = _REPO / "data/seed/yml/pilot100_manifest.tsv"
GT_PATH = _REPO / "data/seed/groundtruth" / "gt_google_robot.json"
GT_CONSOLIDATED = GT_PATH
MEDIA_DIR = _REPO / "data/results/render/google_robot/pilot40_media"
RENDER_DIR = _REPO / "data/results/render/google_robot"
TILE_DIR = _REPO / "data/results/visualize/google_pose_groups_12"
MULTITILE_IMG_DIR = _REPO / "data/results/visualize/google_robot/pose_multitile_gt_pilot40"
PAIRWISE_IMG_DIR = _REPO / "data/results/visualize/google_robot/pose_pairwise_pilot40"
MOTION_MANIFEST = MEDIA_DIR / "manifest_pilot40.json"
MOTION_PAIRWISE_DIR = _REPO / "data/results/verify/samples/motion_gt_neg_pairwise_pilot40"
TOPK_GRID_DIR = _REPO / "data/results/visualize/google_robot/pose_topk_gemini"

RESULT_CFG_DIR = _REPO / "data/results/motion_configs" / ROBOT / "exp"
VERIFY_EXP_DIR = _REPO / "data/results/verify" / ROBOT / "exp"
HTML_EXP_DIR = _REPO / "data/results/html" / ROBOT

# Legacy flat paths (pre-refactor)
LEGACY_VERIFY_DIR = _REPO / "data/results/verify" / ROBOT
LEGACY_CFG = _REPO / "data/results/motion_configs" / ROBOT / "motion_configs_pilot40_mobile.json"

N_CUES = 90
DEFAULT_GEN_TAG = "mobile-map"
DEFAULT_VERIFY_TAG = "gemini-2.5-flash"

EXPERIMENT_TITLES: dict[str, str] = {
    "1": "Pose generation vs human GT (mobile map)",
    "2": "Pose verify + regenerate — VLM (PNG)",
    "3": "Pose verify + regenerate — text",
    "4": "Pose pairwise (VLM) — 2-way",
    "5": "Pose pairwise (VLM) — multitile grid 6",
    "6": "Pose pairwise (VLM) — multitile grid 12",
    "7": "Movement generation vs component GT",
    "8": "Movement verify + regenerate — VLM (MP4)",
    "9": "Movement verify + regenerate — text",
    "10": "Movement pairwise (VLM — MP4)",
}

GENERATION_EXPS = frozenset({"1", "7"})
CONFIG_INPUT_EXPS: dict[str, str] = {
    "1": "1",
    "2": "1",
    "3": "1",
    "7": "7",
    "8": "7",
    "9": "7",
    "10": "7",
}

PROMPT_LEGACY_MAP: dict[str, str] = {
    "1": "prompt_generation_pose_component.txt",
    "2": "prompt_verify_pose_vlm_component.txt",
    "3": "prompt_verify_pose_text_component.txt",
    "4": "prompt_compare_pose_vlm_component.txt",
    "5": "prompt_pick_pose_topk_grid.txt",
    "6": "prompt_compare_pose_vlm_component.txt",
    "7": "prompt_generation_movement_component.txt",
    "8": "prompt_verify_movement_vlm_component.txt",
    "9": "prompt_verify_movement_text_component.txt",
    "10": "prompt_compare_movement_vlm_component.txt",
}


def prompt_exp_path(exp_id: str | int) -> Path:
    return PROMPT_EXP_DIR / f"prompt_exp{int(exp_id)}.txt"


def model_to_tag(model: str) -> str:
    m = model.strip()
    low = m.lower()
    if "mobile-map" in low or low == "map":
        return DEFAULT_GEN_TAG
    if "gemini" in low:
        return m.replace("/", "-")
    if "32b" in low:
        return "qwen32b"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", m)


def result_config_path(exp_id: str | int, model_tag: str) -> Path:
    return RESULT_CFG_DIR / f"result_exp{int(exp_id)}_{model_tag}.json"


def verify_result_path(exp_id: str | int, model_tag: str) -> Path:
    return VERIFY_EXP_DIR / f"result_exp{int(exp_id)}_{model_tag}.json"


def score_result_path(exp_id: str | int, model_tag: str) -> Path:
    return VERIFY_EXP_DIR / f"score_exp{int(exp_id)}_{model_tag}.json"


def html_result_path(exp_id: str | int, model_tag: str) -> Path:
    return HTML_EXP_DIR / f"exp{int(exp_id)}_{model_tag}.html"


def config_for_experiment(exp_id: str | int, model_tag: str) -> Path:
    eid = str(exp_id)
    src = CONFIG_INPUT_EXPS.get(eid)
    if src:
        return result_config_path(src, model_tag)
    if eid in {"4", "5", "6"}:
        return result_config_path("1", model_tag)
    raise KeyError(f"No config input for experiment {eid}")


def manifest90_cue_names() -> list[str]:
    if not MANIFEST_TSV.is_file():
        return []
    out: list[str] = []
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        out.append(parts[3].strip())
    return out


def manifest90_cue_indices() -> dict[str, int]:
    gt = load_gt_by_cue() if GT_PATH.is_file() else {}
    out: dict[str, int] = {}
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        cue = parts[3].strip()
        idx_raw = parts[2].strip()
        if idx_raw.isdigit():
            out[cue] = int(idx_raw)
        elif cue in gt:
            out[cue] = int(gt[cue]["cue_idx"])
    return out


def load_gt_rows() -> list[dict[str, Any]]:
    if not GT_PATH.is_file():
        return []
    data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return list(data.get("rows") or [])


def load_gt_by_cue() -> dict[str, dict[str, Any]]:
    return {str(r["cue"]): r for r in load_gt_rows() if r.get("cue")}


def pilot40_cue_names() -> list[str]:
    """Backward-compat alias — full suite uses manifest90 (90 cues)."""
    names = manifest90_cue_names()
    if names:
        return names
    if not SHOTS.is_file():
        return []
    return [str(r["cue"]) for r in load_config_list(SHOTS) if r.get("cue")]


def row_generation_done(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if row.get("generation_valid") is True:
        return True
    if row.get("generation_valid") is None and row.get("movements"):
        return True
    return False


def upsert_config_row(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_config_list(path)
    cue = row.get("cue")
    rows = [r for r in rows if r.get("cue") != cue]
    rows.append(row)
    rows.sort(key=lambda r: int(r.get("idx", 0)))
    save_json(path, rows)
    return rows


def load_config_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
