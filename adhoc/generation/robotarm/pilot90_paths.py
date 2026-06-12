"""Canonical paths for pilot-90 (manipulator) — prompts, GT, per-model results."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parents[2]
ROBOT = "manipulator"

PROMPT_EXP_DIR = _REPO / "data/seed/prompt" / ROBOT / "exp"
GT_PATH = _REPO / "data/seed/groundtruth" / f"gt_{ROBOT}.json"
SHOTS = _REPO / "data/seed/shots" / ROBOT / "shot_configs_v19_sophisticated.json"
MANIFEST_TSV = _REPO / "data/seed/yml/pilot100_manifest.tsv"

RESULT_CFG_DIR = _REPO / "data/results/motion_configs" / ROBOT / "exp"
VERIFY_EXP_DIR = _REPO / "data/results/verify" / ROBOT / "exp"
HTML_EXP_DIR = _REPO / "data/results/html"

TILE_DIR = _REPO / "data/results/visualize/pose_groups_12"
TILE_PICK = _REPO / "data/results/verify/pose_tile_pick_by_group.json"
PAIRWISE_IMG_DIR = _REPO / "data/results/visualize/pose_pairwise_12_pilot90"
MULTITILE_IMG_DIR = _REPO / "data/results/visualize/pose_multitile_gt_pilot90"
MOTION_MANIFEST = _REPO / "data/results/render/manipulator/motion_vlm_verify_pilot90/manifest_pilot90.json"
MOTION_PAIRWISE_DIR = _REPO / "data/results/verify/samples/motion_gt_neg_pairwise_pilot90"

N_CUES = 90

# Legacy aliases (deprecated — use result_config_path / verify_result_path)
POSE_CFG_LEGACY = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
CONSOLIDATED_LEGACY = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
MOTION_COMPONENT_GT_LEGACY = _REPO / "data/results/verify/pilot40_motion_component_gt.json"

EXPERIMENT_TITLES: dict[str, str] = {
    "1": "Pose generation vs human GT",
    "2": "Pose verify + regenerate — VLM (tile)",
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


def prompt_exp_path(exp_id: str | int) -> Path:
    return PROMPT_EXP_DIR / f"prompt_exp{int(exp_id)}.txt"


def model_to_tag(model: str) -> str:
    m = model.strip()
    low = m.lower()
    if "gemini" in low:
        return m.replace("/", "-")
    if "32b" in low:
        return "qwen32b"
    if "7b" in low:
        return "qwen7b"
    if "3b" in low:
        return "qwen3b"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", m)


def result_config_path(exp_id: str | int, model_tag: str) -> Path:
    """Per-model motion config list (exp 1, 7) — JSON array of 90 cue dicts."""
    return RESULT_CFG_DIR / f"result_exp{int(exp_id)}_{model_tag}.json"


def verify_result_path(exp_id: str | int, model_tag: str) -> Path:
    """Per-model verify / score output (exp 2–6, 8–10, and score sidecars)."""
    return VERIFY_EXP_DIR / f"result_exp{int(exp_id)}_{model_tag}.json"


def score_result_path(exp_id: str | int, model_tag: str) -> Path:
    return VERIFY_EXP_DIR / f"score_exp{int(exp_id)}_{model_tag}.json"


def html_result_path(exp_id: str | int, model_tag: str) -> Path:
    """Per-experiment review HTML, e.g. data/results/html/exp2_qwen32b.html."""
    return HTML_EXP_DIR / f"exp{int(exp_id)}_{model_tag}.html"


def config_for_experiment(exp_id: str | int, model_tag: str) -> Path:
    """Motion config JSON used as input for an experiment step."""
    eid = str(exp_id)
    src = CONFIG_INPUT_EXPS.get(eid)
    if src:
        return result_config_path(src, model_tag)
    if eid in {"4", "5", "6"}:
        return result_config_path("1", model_tag)
    raise KeyError(f"No config input mapping for experiment {eid}")


def manifest90_cue_names() -> list[str]:
    out: list[str] = []
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        out.append(parts[3].strip())
    return out


def manifest90_cue_indices() -> dict[str, int]:
    """cue → cue_idx (manifest idx column, else GT file)."""
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
    data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return list(data.get("rows") or [])


def load_gt_by_cue() -> dict[str, dict[str, Any]]:
    return {str(r["cue"]): r for r in load_gt_rows() if r.get("cue")}


def load_config_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_config_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_config_row(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_config_list(path)
    cue = row.get("cue")
    rows = [r for r in rows if r.get("cue") != cue]
    rows.append(row)
    rows.sort(key=lambda r: int(r.get("idx", 0)))
    save_config_list(path, rows)
    return rows


def row_generation_done(row: dict[str, Any] | None) -> bool:
    """True when a cue row finished generation (valid output, or legacy row without flag)."""
    if not row:
        return False
    if row.get("generation_valid") is True:
        return True
    if row.get("generation_valid") is None and row.get("movements"):
        return True
    return False
