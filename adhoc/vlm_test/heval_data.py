"""
Load evaluation instances aligned with humeaneval PPTX (binary_classification, compare_baseline).
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEV_R = Path(__file__).resolve().parents[2]
_HE = _DEV_R / "adhoc" / "humaneval"
if str(_HE) not in sys.path:
    sys.path.insert(0, str(_HE))

# compare_baseline shuffle (same as PPTX)
import hashlib  # noqa: E402

from _pptx_lib import (  # noqa: E402
    MANIP_CFG,
    apply_sample_n_iiwa_fewshot_first,
    latest_gif_in_dir,
    latest_iiwa_direct_baseline_gif,
    load_binary_for_robot,
    load_catalog_gifs,
    load_manipulator_product_list,
    parse_robot,
    resolve_binary_item_gif,
)

BINARY_Q = "Does the shown cue match this motion? Answer with only Yes or No."
COMPARE_Q = (
    "Select all options (A–D) whose motion best matches the given cue. "
    "List every letter that applies, separated by commas (e.g. A, C, D). "
    "The sophisticated reference option must be included in any good answer, but you may add others you believe fit."
)


def _per_slide_shuffle_seed(cue: str, idx: int, sub: str) -> int:
    h = hashlib.md5(f"{sub}|{idx}|{cue}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def _iiwa_combined_shuffle_seed(position_seed: int, cue: str, idx: int, sub: str) -> int:
    base = int(position_seed) & 0x7FFFFFFF
    mix = _per_slide_shuffle_seed(cue, idx, sub)
    return (base * 0x9E3779B9 + mix) & 0x7FFFFFFF


@dataclass
class HevalVLMInstance:
    """One VLM call worth of data (PPTX-aligned)."""

    instance_id: str
    task: str
    robot: str
    prompt: str
    media: list[tuple[str, Path, str]]  # (role, path, mime)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _config_path_for_idx(subtest: str) -> Path:
    return Path(MANIP_CFG[subtest]["soph"])


def _sim_robot_name(robot: str) -> str:
    """Name passed to MotionGenerator / prepare_test_media (arm sim, not PPTX 'manipulator' label)."""
    rk = (robot or "").strip().lower()
    if rk == "manipulator":
        return "IIWA"
    if rk == "tiago":
        return "Tiago"
    return "IIWA"


def _config_path_for_sample(robot: str, sub: str) -> Path:
    rk = (robot or "").strip().lower()
    if rk == "manipulator":
        return _config_path_for_idx(sub)
    if rk == "tiago":
        return (
            _DEV_R
            / "data"
            / "results"
            / "motion_configs"
            / "google_robot"
            / "motion_configs_19_mobile.json"
        )
    return _config_path_for_idx(sub)


def _binary_item_to_sample_dict(it: dict, robot: str, order: int) -> dict[str, Any]:
    """Adapter for testset_utils.prepare_test_media (needs config_path for sim-heavy types)."""
    sub = (it.get("testset") or it.get("subset") or "iconic").lower()
    if sub not in ("iconic", "contextual"):
        sub = "iconic"
    idx = int(it.get("config_idx", it.get("idx", it.get("cue_index", 0))))
    cfg_path = _config_path_for_sample(robot, sub)
    g = resolve_binary_item_gif(it, prefer_no_reasoning=(robot == "manipulator"))
    cue = (it.get("source_cue") or it.get("shown_cue") or "").strip() or (it.get("cue") or "")
    return {
        "sample_id": f"{robot}_bin_{order}_{sub}_c{idx:02d}",
        "testset": sub,
        "cue_idx": idx,
        "cue": cue,
        "gif_path": str(g) if g else "",
        "config_path": str(cfg_path),
        "sim_robot": _sim_robot_name(robot),
        "meta": {"item": it},
    }


def load_binary_instances(
    robot: str = "all",
    sample_n: int | None = None,
    seed: int = 17,
    first_n: int | None = None,
) -> list[HevalVLMInstance]:
    rlist = parse_robot(robot)
    out: list[HevalVLMInstance] = []
    for rk in rlist:
        items = list(load_binary_for_robot(rk))
        if sample_n is not None and len(items) > sample_n:
            rng = random.Random(seed)
            items = list(items)
            rng.shuffle(items)
            items = items[: sample_n]
        if first_n is not None:
            items = items[: int(first_n)]
        for o, it in enumerate(items, start=1):
            sub = (it.get("testset") or "iconic").lower()
            shown = (it.get("shown_cue") or it.get("source_cue") or "").strip()
            gt = (it.get("ground_truth") or "").strip().lower()
            g = resolve_binary_item_gif(it, prefer_no_reasoning=(rk == "manipulator"))
            if not g or not g.is_file():
                continue
            prompt = f'Shown cue: "{shown}"\n\n{BINARY_Q}'
            hid = f"binary_{rk}_{sub}_c{it.get('config_idx', it.get('idx', o))}_{o}"
            out.append(
                HevalVLMInstance(
                    instance_id=hid,
                    task="binary_classification",
                    robot=rk,
                    prompt=prompt,
                    media=[("motion", g, "image/gif")],
                    ground_truth={"kind": "yesno", "answer": gt},
                    meta={"_sample": _binary_item_to_sample_dict(it, rk, o), "_item": it},
                )
            )
    return out


def load_compare_iiwa_instances(
    sample_n: int | None = None,
    position_seed: int = 20260424,
) -> list[HevalVLMInstance]:
    rows = apply_sample_n_iiwa_fewshot_first(
        load_manipulator_product_list(motion="soph"), sample_n, position_seed
    )
    letters = ("A", "B", "C", "D")
    out: list[HevalVLMInstance] = []
    for o, r in enumerate(rows, start=1):
        spec = MANIP_CFG[r.subtest]
        cue = r.cue
        idx = int(r.idx)
        g_soph = r.gif
        g_nr = latest_gif_in_dir(spec["nr_gif"], cue)
        g_joint = latest_iiwa_direct_baseline_gif(cue, idx, r.subtest, joint=True)
        g_xyz = latest_iiwa_direct_baseline_gif(cue, idx, r.subtest, joint=False)
        panels: list[tuple[str, Path | None]] = [
            ("soph", g_soph),
            ("nr", g_nr),
            ("joint", g_joint),
            ("xyz", g_xyz),
        ]
        if not all(p and p.is_file() for _, p in panels):
            continue
        sh = _iiwa_combined_shuffle_seed(position_seed, cue, idx, r.subtest)
        rng = random.Random(int(sh))
        order = list(panels)
        rng.shuffle(order)
        soph_letter = next(letters[i] for i, (k, _) in enumerate(order) if k == "soph")
        desc = (r.label or "")[:500]
        prompt = f'Cue: "{cue}"\n{desc}\n\n{COMPARE_Q}'
        media: list[tuple[str, Path, str]] = []
        for i, (kind, g) in enumerate(order):
            if g and g.is_file():
                media.append((f"option_{letters[i]}_{kind}", g, "image/gif"))
        if len(media) != 4:
            continue
        out.append(
            HevalVLMInstance(
                instance_id=f"compare_iiwa_{r.subtest}_c{idx:02d}_{o}",
                task="compare_baseline",
                robot="manipulator",
                prompt=prompt,
                media=media,
                ground_truth={
                    "kind": "multiselect4",
                    "soph_letter": soph_letter,
                    "subtest": r.subtest,
                    "idx": r.idx,
                    "cue": r.cue,
                },
            )
        )
    return out


def load_compare_non_iiwa_instances(
    robot: str,
    sample_n: int | None = None,
    seed: int = 17,
) -> list[HevalVLMInstance]:
    """Left column = current only (future baseline column empty in PPTX). Single GIF task."""
    rk = (robot or "").strip().lower()
    if rk not in ("tiago", "quadruped"):
        return []
    rows = list(load_catalog_gifs(rk))
    if sample_n is not None and len(rows) > sample_n:
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[: sample_n]
    out: list[HevalVLMInstance] = []
    for o, r in enumerate(rows, start=1):
        g = r.gif
        if not g or not g.is_file():
            continue
        prompt = f'Cue: "{r.cue}"\n{(r.label or "")[:500]}\n\nDoes this single motion match the cue for this embodiment? Answer Yes or No only.'
        out.append(
            HevalVLMInstance(
                instance_id=f"compare_{rk}_c{r.idx}_{o}",
                task="compare_baseline",
                robot=rk,
                prompt=prompt,
                media=[("current_render", g, "image/gif")],
                ground_truth={
                    "kind": "open",
                    "subtest": r.subtest,
                    "idx": r.idx,
                    "cue": r.cue,
                },
            )
        )
    return out


def load_compare_instances(
    robot: str = "all",
    sample_n: int | None = None,
    position_seed: int = 20260424,
    seed: int = 17,
) -> list[HevalVLMInstance]:
    rlist = parse_robot(robot)
    out: list[HevalVLMInstance] = []
    if "manipulator" in rlist:
        out.extend(load_compare_iiwa_instances(sample_n, position_seed=position_seed))
    for rk in rlist:
        if rk in ("tiago", "quadruped"):
            out.extend(load_compare_non_iiwa_instances(rk, sample_n, seed=seed))
    return out
