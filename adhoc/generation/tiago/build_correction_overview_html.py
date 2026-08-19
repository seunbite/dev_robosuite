#!/usr/bin/env python3
"""Overview HTML for all 90 exp1 correction round-1 results with outcome badges."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    config_snapshot_path,
    correction_stem,
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    load_config_list,
    load_correction_record,
)
from render_paths import (  # noqa: E402
    gt_render_role,
    render_gif_path,
    render_media_paths,
    render_role_for_round,
)
from correction_prompts import correction_indicates_verified_ok  # noqa: E402
from parse_pose_gt_mobile import pose_generation_correct_any_mobile, pose_gt_display  # noqa: E402
from pilot40_paths import load_gt_by_cue, score_result_path  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
OUT_DIR = _REPO / "data/results/visualize/tiago/correction/exp1"


def _esc(s: object) -> str:
    return html.escape(str(s))


def _rel(out: Path, p: Path) -> str:
    if not p.is_file():
        return ""
    rel = os.path.relpath(p, out.parent).replace("\\", "/")
    # Bust browser cache when GIFs are re-rendered in place.
    mtime = int(p.stat().st_mtime)
    return _esc(f"{rel}?v={mtime}")


def _generation_correct_by_cue(model_tag: str) -> dict[str, bool | None]:
    score_path = score_result_path(1, model_tag)
    out: dict[str, bool | None] = {}
    if score_path.is_file():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        for row in score.get("rows") or []:
            cue = str(row.get("cue", ""))
            if cue:
                out[cue] = bool(row.get("generation_correct"))
    gt_by_cue = load_gt_by_cue()
    cfg_path = default_config_path("tiago", model_tag, "1")  # type: ignore[arg-type]
    for row in load_config_list(cfg_path):
        cue = str(row.get("cue", ""))
        if not cue or cue in out:
            continue
        gt_row = gt_by_cue.get(cue) or {}
        ev = str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or "")
        if ev:
            out[cue] = pose_generation_correct_any_mobile(row, ev)
        else:
            out[cue] = None
    return out


def effective_action(action: str | None, correction: dict | None) -> str:
    act = action or "pending"
    if act == "corrected" and correction and correction_indicates_verified_ok(correction):
        return "verified_ok"
    return act


def gen_badge(gen_correct: bool | None) -> tuple[str, str]:
    if gen_correct is True:
        return "True", "badge-true"
    if gen_correct is False:
        return "False", "badge-false"
    return "—", "badge-unknown"


def corr_badge(action: str | None) -> tuple[str, str]:
    act = action or "pending"
    if act == "verified_ok":
        return "Verified", "badge-verified"
    if act == "corrected":
        return "Edited", "badge-edited"
    if act == "pending":
        return "Pending", "badge-pending"
    return _esc(act), "badge-other"


def _pose_steps(row: dict | None) -> list:
    return [s for s in (row or {}).get("movements", []) if s.get("type") == "pose"]


def _motion_steps(row: dict | None) -> list:
    return [
        s
        for s in (row or {}).get("movements", [])
        if s.get("type") in ("movement", "path")
    ]


def correction_edit_group(
    orig_row: dict | None,
    corr_row: dict | None,
    action: str | None,
) -> str:
    """Classify corr1 edit: verified | pose | movement | both | pending."""
    act = action or "pending"
    if act == "verified_ok":
        return "verified"
    if act != "corrected":
        return "pending"
    pose_changed = json.dumps(_pose_steps(orig_row), sort_keys=True) != json.dumps(
        _pose_steps(corr_row), sort_keys=True
    )
    motion_changed = json.dumps(_motion_steps(orig_row), sort_keys=True) != json.dumps(
        _motion_steps(corr_row), sort_keys=True
    )
    if pose_changed and motion_changed:
        return "both"
    if pose_changed:
        return "pose"
    if motion_changed:
        return "movement"
    return "both"


def _is_q2_na(q2: str | None) -> bool:
    s = str(q2 or "").strip().lower()
    return s in ("n/a", "na", "n.a.", "n.a", "none") or s.startswith("n/a")


EDIT_GROUP_ORDER: list[tuple[str, str]] = [
    ("verified", "1. 괜찮다고 한 경우 (Q2 n/a)"),
    ("pose", "2. Pose 수정"),
    ("movement", "3. Movement 수정"),
    ("both", "4. Pose / Movement 둘 다 수정"),
    ("pending", "5. 아직 진행 중"),
]

VISUAL_GROUP_ORDER: list[tuple[str, str]] = [
    ("oo", "1. o / o"),
    ("ox", "2. o / x"),
    ("xo", "3. x / o"),
    ("xx", "4. x / x"),
    ("other", "5. 기타 / 미분류"),
    ("pending", "6. corr1 미완료"),
]


def _load_visual_labels(model_tag: str) -> dict[str, dict[str, str]]:
    path = OUT_DIR / model_tag / "visual_outcome_labels.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): v for k, v in raw.items()}


def _visual_symbols_from_file(gen_correct: bool | None, action: str | None) -> tuple[str, str]:
    """Fallback visual rating from VLM gen/corr badges when manual label missing."""
    g = "o" if gen_correct is True else ("x" if gen_correct is False else "?")
    act = action or "pending"
    if act == "verified_ok":
        c = "o"
    elif act == "corrected":
        c = "x"
    else:
        c = ""
    return g, c


def visual_grid_group(orig_sym: str, corr_sym: str) -> str:
    """Manual visual rating grid: original × corr1."""
    if orig_sym == "o" and corr_sym == "o":
        return "oo"
    if orig_sym == "o" and corr_sym == "x":
        return "ox"
    if orig_sym == "x" and corr_sym == "o":
        return "xo"
    if orig_sym == "x" and corr_sym == "x":
        return "xx"
    return "other"


def visual_outcome_group(orig_sym: str, corr_sym: str, *, verified_ok: bool) -> str:
    if (orig_sym == "x" and corr_sym == "o") or (orig_sym == "o" and corr_sym == "oo"):
        return "fixed_improved"
    if orig_sym == "o" and corr_sym in ("x", "xx"):
        return "made_worse"
    if orig_sym == "o" and corr_sym == "o":
        return "passed" if verified_ok else "unnecessary_regen"
    if orig_sym == "x" and corr_sym in ("x", "xx"):
        return "fix_failed"
    if not corr_sym:
        return "other"
    return "other"


def visual_badge_html(orig_sym: str, corr_sym: str) -> str:
    o_cls = "badge-true" if orig_sym == "o" else "badge-false" if orig_sym == "x" else "badge-unknown"
    if corr_sym == "o":
        c_cls, c_lbl = "badge-verified", "o"
    elif corr_sym == "oo":
        c_cls, c_lbl = "badge-verified", "oo"
    elif corr_sym == "x":
        c_cls, c_lbl = "badge-edited", "x"
    elif corr_sym == "xx":
        c_cls, c_lbl = "badge-edited", "xx"
    else:
        c_cls, c_lbl = "badge-pending", "—"
    return (
        f'<span class="badge {o_cls}" title="Original motion">{_esc(orig_sym)}</span>'
        f'<span class="badge {c_cls}" title="Correction motion">{_esc(c_lbl)}</span>'
    )


def _config_display_row(row: dict | None) -> dict:
    if not row:
        return {}
    keys = (
        "idx",
        "cue",
        "description",
        "movements",
        "state",
    )
    return {k: row[k] for k in keys if row.get(k) is not None}


def _display_q_text(text: str | None, *, q_num: int) -> str:
    """Strip echoed prompt prefix (e.g. 'Correction Q1:') from VLM answer."""
    if not text:
        return ""
    s = str(text).strip()
    for prefix in (
        f"Correction Q{q_num}:",
        f"correction q{q_num}:",
        f"Q{q_num}:",
        f"q{q_num}:",
    ):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :].strip()
            break
    return s


def _config_json_html(row: dict | None, *, summary: str = "Config") -> str:
    payload = _config_display_row(row)
    if not payload:
        return '<div class="cfg-missing">Config missing</div>'
    body = _esc(json.dumps(payload, indent=2, ensure_ascii=False))
    return (
        f'<details class="cfg-toggle">'
        f"<summary>{_esc(summary)}</summary>"
        f"<pre>{body}</pre>"
        f"</details>"
    )


def _load_corr1_row(model_tag: str, exp_id: str, cue: str) -> dict | None:
    snap = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
    if not snap.is_file():
        return None
    data = json.loads(snap.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def _load_snapshot_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def _new_render_round_snapshot_path(model_tag: str, exp_id: str, cue: str, round_n: int) -> Path:
    if round_n == 1:
        return config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
    return config_snapshot_path("tiago", exp_id, model_tag, cue, round_n).with_name(
        f"{correction_stem(round_n)}_new_render_config.json"
    )


def _render_paths(
    idx: int,
    cue: str,
    model_tag: str,
    exp_id: str,
    *,
    round_n: int | None,
    role: str | None = None,
) -> tuple[Path, Path]:
    if round_n is None:
        r = role or "gen"
    else:
        r = role or render_role_for_round(round_n)
    gif, mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, r)
    return gif, mp4


def _gt_config_path(model_tag: str, exp_id: str, cue: str, *, variant: str = "gt") -> Path:
    name = "gt_config.json" if variant == "gt" else f"gt_{variant}_config.json"
    return (
        _REPO
        / "data/results/motion_configs/tiago/exp/correction"
        / model_tag
        / f"exp{exp_id}"
        / cue
        / name
    )


def _gt_render_paths(
    idx: int, cue: str, model_tag: str, exp_id: str, *, variant: str = "gt"
) -> tuple[Path, Path]:
    role = gt_render_role(variant, round_n=0)
    gif, mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, role)
    return gif, mp4


_GT_RENDER_VARIANTS: tuple[tuple[str, str], ...] = (
    ("gt", "GT render (z)"),
    ("y", "GT render (elbow y)"),
)


def _gt_correction_config_path(
    model_tag: str, exp_id: str, cue: str, *, variant: str = "gt", round_n: int = 1
) -> Path:
    stem = correction_stem(round_n)
    if variant == "gt":
        return _gt_config_path(model_tag, exp_id, cue).with_name(f"gt_{stem}_config.json")
    return _gt_config_path(model_tag, exp_id, cue, variant=variant).with_name(
        f"gt_{variant}_{stem}_config.json"
    )


def _gt_correction_render_paths(
    idx: int,
    cue: str,
    model_tag: str,
    exp_id: str,
    *,
    variant: str = "gt",
    round_n: int = 1,
) -> tuple[Path, Path]:
    role = gt_render_role(variant, round_n=round_n)
    gif, mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, role)
    return gif, mp4


def _gt_correction_record_name(variant: str = "gt", round_n: int = 1) -> str:
    stem = correction_stem(round_n)
    if variant == "gt":
        return f"gt_{stem}.json"
    return f"gt_{variant}_{stem}.json"


def _sophistication_record_path(model_tag: str, exp_id: str, cue: str) -> Path:
    return (
        _REPO
        / "data/results/verify/tiago/sophistication"
        / f"exp{exp_id}"
        / model_tag
        / cue
        / "orig_new_render_sophistication.json"
    )


def _load_sophistication_record(model_tag: str, exp_id: str, cue: str) -> dict | None:
    path = _sophistication_record_path(model_tag, exp_id, cue)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _corr_round_gif_exists(
    idx: int,
    cue: str,
    model_tag: str,
    exp_id: str,
    round_n: int,
) -> bool:
    gif, _ = _render_paths(idx, cue, model_tag, exp_id, round_n=round_n)
    return gif.is_file()


def _row_with_correction_qs(
    row: dict | None,
    q1: str | None,
    q2: str | None,
) -> dict | None:
    if not row:
        return None
    out = dict(row)
    if q1:
        out["correction_q1"] = q1
    if q2 is not None:
        out["correction_q2"] = q2
    return out


def build_overview_html(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    out_path: Path | None = None,
    only_cue: str | None = None,
) -> Path:
    out = out_path or (OUT_DIR / model_tag / "correction_overview.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    gen_ok = _generation_correct_by_cue(model_tag)
    cfg_rows = sorted(
        load_config_list(default_config_path("tiago", model_tag, exp_id)),  # type: ignore[arg-type]
        key=lambda r: int(r.get("idx", 0)),
    )
    if only_cue:
        cfg_rows = [r for r in cfg_rows if str(r.get("cue", "")) == only_cue]
        if not cfg_rows:
            raise ValueError(f"cue not found: {only_cue}")
    gt_by_cue = load_gt_by_cue()

    visual_labels = _load_visual_labels(model_tag)
    counts: dict[str, int] = {
        "gen_true": 0,
        "gen_false": 0,
        "gen_unknown": 0,
        "corr_edited": 0,
        "corr_verified": 0,
        "corr_pending": 0,
        "corr_other": 0,
    }
    visual_counts: dict[str, int] = {k: 0 for k, _ in VISUAL_GROUP_ORDER}
    cards_by_group: dict[str, list[str]] = {k: [] for k, _ in VISUAL_GROUP_ORDER}

    for row in cfg_rows:
        idx = int(row.get("idx", -1))
        cue = str(row.get("cue", ""))
        gen_correct = gen_ok.get(cue)
        gt_row = gt_by_cue.get(cue) or {}
        pose_gt = pose_gt_display(
            bool(gt_row.get("pose_any")),
            list(gt_row.get("pose_options") or []),
            str(gt_row.get("pose_gt") or gt_row.get("groundtruth") or ""),
        )

        cue_dir = correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
        rec_path = correction_record_path(cue_dir, 1)
        corr1_done = _corr_round_gif_exists(idx, cue, model_tag, exp_id, 1)
        action: str | None = None
        q1 = q2 = None
        corr: dict = {}
        if rec_path.is_file() and corr1_done:
            rec = load_correction_record(cue_dir, 1)
            action = effective_action(str(rec.get("action") or ""), rec.get("correction"))
            corr = rec.get("correction") or {}
            q1 = corr.get("correction_q1")
            q2 = corr.get("correction_q2")
            if not q1 and corr.get("inner_reasoning"):
                from build_correction_evolution_html import _correction_qs  # noqa: WPS433

                q1, q2 = _correction_qs(corr)
        else:
            action = "pending"

        gen_label, gen_cls = gen_badge(gen_correct)
        corr_label, corr_cls = corr_badge(action)
        if gen_cls == "badge-true":
            counts["gen_true"] += 1
        elif gen_cls == "badge-false":
            counts["gen_false"] += 1
        else:
            counts["gen_unknown"] += 1
        if corr_cls == "badge-edited":
            counts["corr_edited"] += 1
        elif corr_cls == "badge-verified":
            counts["corr_verified"] += 1
        elif corr_cls == "badge-pending":
            counts["corr_pending"] += 1
        else:
            counts["corr_other"] += 1

        manual = visual_labels.get(str(idx))
        visual_note = ""
        if manual:
            v_orig = manual.get("orig", "?")
            v_corr = manual.get("corr", "")
            visual_note = str(manual.get("note") or "").strip()
        else:
            v_orig, v_corr = _visual_symbols_from_file(gen_correct, action)

        corr_row = _load_corr1_row(model_tag, exp_id, cue) if corr1_done else None
        if not corr1_done:
            visual_group = "pending"
        else:
            visual_group = visual_grid_group(v_orig, v_corr)
        visual_counts[visual_group] = visual_counts.get(visual_group, 0) + 1

        soph_record = _load_sophistication_record(model_tag, exp_id, cue)
        new_round_rows: dict[int, dict | None] = {
            1: corr_row,
            2: _load_snapshot_row(_new_render_round_snapshot_path(model_tag, exp_id, cue, 2)),
            3: _load_snapshot_row(_new_render_round_snapshot_path(model_tag, exp_id, cue, 3)),
        }

        vlm_html = (
            f'<span class="badge {gen_cls}" title="VLM gen GT">{_esc(gen_label)}</span>'
            f'<span class="badge {corr_cls}" title="VLM corr">{_esc(corr_label)}</span>'
        )
        outcome_html = (
            f'<div class="badge-row"><span class="badge-label">Visual</span>{visual_badge_html(v_orig, v_corr)}</div>'
            f'<div class="badge-row"><span class="badge-label">VLM</span>{vlm_html}</div>'
        )

        def _fig(
            title: str,
            gif: Path,
            mp4: Path,
            config_row: dict | None,
            *,
            fig_q1: str | None = None,
            fig_q2: str | None = None,
        ) -> str:
            gif_r = _rel(out, gif)
            if gif_r:
                body = f'<img src="{gif_r}" loading="lazy" alt="{_esc(title)}">'
            else:
                body = '<div class="missing">GIF missing</div>'
            mp4_r = _rel(out, mp4)
            if mp4_r:
                body += f'<p class="mp4"><a href="{mp4_r}">MP4</a></p>'
            body += _config_json_html(config_row, summary="Config")
            if fig_q1:
                body += f'<div class="vq"><b>Q1:</b> {_esc(_display_q_text(fig_q1, q_num=1))}</div>'
            if fig_q2 is not None:
                body += f'<div class="vq"><b>Q2:</b> {_esc(_display_q_text(fig_q2, q_num=2))}</div>'
            return f'<figure><figcaption>{_esc(title)}</figcaption>{body}</figure>'

        q_html = ""
        if soph_record:
            s_q1 = soph_record.get("sophistication_q1")
            s_q2 = soph_record.get("sophistication_q2")
            if s_q1 or s_q2:
                q_html += (
                    f'<div class="vq"><b>Sophistication Q1:</b> {_esc(s_q1 or "")}</div>'
                    f'<div class="vq"><b>Sophistication Q2:</b> {_esc(s_q2 or "")}</div>'
                )

        figs = [
            _fig(
                "Generation",
                *_render_paths(idx, cue, model_tag, exp_id, round_n=0),
                row,
            )
        ]
        for round_n in (1, 2, 3):
            round_row = new_round_rows[round_n]
            round_gif, round_mp4 = _render_paths(
                idx, cue, model_tag, exp_id, round_n=round_n
            )
            if not round_gif.is_file():
                continue
            fig_q1 = q1 if round_n == 1 else None
            fig_q2 = q2 if round_n == 1 else None
            figs.append(
                _fig(
                    f"Correction {round_n}",
                    round_gif,
                    round_mp4,
                    round_row,
                    fig_q1=fig_q1,
                    fig_q2=fig_q2,
                )
            )
        if soph_record:
            soph_row = soph_record.get("output_row")
            soph_gif, soph_mp4, _ = render_media_paths(
                "tiago", model_tag, exp_id, idx, cue, "soph"
            )
            if not soph_gif.is_file():
                media = soph_record.get("rendered_media") or {}
                soph_gif = Path(str(media.get("gif") or ""))
                soph_mp4 = Path(str(media.get("mp4") or ""))
            figs.append(_fig("Sophistication", soph_gif, soph_mp4, soph_row))

        gt_cfg_path = _gt_config_path(model_tag, exp_id, cue)
        if gt_cfg_path.is_file():
            gt_row = _load_snapshot_row(gt_cfg_path)
            gt_gif, gt_mp4 = _gt_render_paths(idx, cue, model_tag, exp_id)
            figs.append(_fig("GT render (z)", gt_gif, gt_mp4, gt_row))

        for variant, title in _GT_RENDER_VARIANTS:
            if variant == "gt":
                continue
            gt_var_cfg = _gt_config_path(model_tag, exp_id, cue, variant=variant)
            if not gt_var_cfg.is_file():
                continue
            gt_var_row = _load_snapshot_row(gt_var_cfg)
            gt_var_gif, gt_var_mp4 = _gt_render_paths(idx, cue, model_tag, exp_id, variant=variant)
            figs.append(_fig(title, gt_var_gif, gt_var_mp4, gt_var_row))

        gt_corr_cfg = _gt_correction_config_path(model_tag, exp_id, cue)
        if gt_corr_cfg.is_file():
            gt_corr_row = _load_snapshot_row(gt_corr_cfg)
            gt_corr_gif, gt_corr_mp4 = _gt_correction_render_paths(idx, cue, model_tag, exp_id)
            gt_corr_q1 = gt_corr_q2 = None
            gt_corr_rec = (
                correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
                / _gt_correction_record_name("gt")
            )
            if gt_corr_rec.is_file():
                rec = json.loads(gt_corr_rec.read_text(encoding="utf-8"))
                corr = rec.get("correction") or {}
                gt_corr_q1 = corr.get("correction_q1")
                gt_corr_q2 = corr.get("correction_q2")
            figs.append(
                _fig(
                    "GT correction (z)",
                    gt_corr_gif,
                    gt_corr_mp4,
                    gt_corr_row,
                    fig_q1=gt_corr_q1,
                    fig_q2=gt_corr_q2,
                )
            )

        for variant, title in _GT_RENDER_VARIANTS:
            if variant == "gt":
                continue
            label = title.removeprefix("GT render ").strip("()")
            for round_n in range(1, 4):
                gt_var_corr_cfg = _gt_correction_config_path(
                    model_tag, exp_id, cue, variant=variant, round_n=round_n
                )
                if not gt_var_corr_cfg.is_file():
                    break
                gt_var_corr_row = _load_snapshot_row(gt_var_corr_cfg)
                gt_var_corr_gif, gt_var_corr_mp4 = _gt_correction_render_paths(
                    idx, cue, model_tag, exp_id, variant=variant, round_n=round_n
                )
                gt_var_corr_q1 = gt_var_corr_q2 = None
                gt_var_corr_rec = (
                    correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
                    / _gt_correction_record_name(variant, round_n=round_n)
                )
                if gt_var_corr_rec.is_file():
                    rec = json.loads(gt_var_corr_rec.read_text(encoding="utf-8"))
                    corr = rec.get("correction") or {}
                    gt_var_corr_q1 = corr.get("correction_q1")
                    gt_var_corr_q2 = corr.get("correction_q2")
                corr_title = (
                    f"GT correction ({label})"
                    if round_n == 1
                    else f"GT correction ({label}) {round_n}"
                )
                figs.append(
                    _fig(
                        corr_title,
                        gt_var_corr_gif,
                        gt_var_corr_mp4,
                        gt_var_corr_row,
                        fig_q1=gt_var_corr_q1,
                        fig_q2=gt_var_corr_q2,
                    )
                )

        meta_parts = [f"<b>GT pose</b>: {_esc(pose_gt)}"]
        if visual_note:
            meta_parts.append(f"<b>Note</b>: {_esc(visual_note)}")
        meta_html = '<div class="meta">' + " · ".join(meta_parts) + "</div>"

        card_html = (
            f'<section class="card" id="g{idx:02d}">'
            f'<div class="head">'
            f'<h2>g{idx:02d} <code>{_esc(cue)}</code></h2>'
            f'<div class="badges">{outcome_html}</div>'
            f"</div>"
            f"{meta_html}"
            f'<div class="media-grid">'
            f'{"".join(figs)}'
            f"</div>"
            f"{q_html}"
            f"</section>"
        )
        cards_by_group.setdefault(visual_group, []).append(card_html)

    sections: list[str] = []
    for group_id, group_title in VISUAL_GROUP_ORDER:
        group_cards = cards_by_group.get(group_id) or []
        if not group_cards:
            continue
        n = len(group_cards)
        sections.append(
            f'<section class="group-section" id="group-{group_id}">'
            f'<h2 class="group-title">{_esc(group_title)} '
            f'<span class="group-count">({n})</span></h2>'
            f'<div class="cards">{"".join(group_cards)}</div>'
            f"</section>"
        )

    n_total = len(cfg_rows)
    legend = (
        '<div class="legend">'
        '<span class="text-muted">Columns:</span> '
        '<b>New render</b> = original config + xyz-trim renderer · '
        '<b>New render (correction 1)</b> = corr1 config + xyz-trim renderer · '
        '<b>New render (correction 2)</b> = corr2 config + xyz-trim renderer · '
        '<b>New render (correction 3)</b> = corr3 config + xyz-trim renderer · '
        '<b>New render (sophistication)</b> = original config + added auxiliary beat (when available) · '
        '<b>GT render (z/y)</b> = manual gt_config.json variants + xyz-trim renderer (when present) · '
        '<b>GT correction</b> = VLM correction on GT render variants (when present)'
        "</div>"
        '<div class="legend">'
        '<span class="text-muted">Groups (챗점):</span> '
        '<b>o/o</b> original OK · corr1 OK · '
        '<b>o/x</b> original OK · corr1 worse · '
        '<b>x/o</b> original bad · corr1 fixed · '
        '<b>x/x</b> original bad · corr1 still bad · '
        '<span class="text-muted">Visual badge = manual rating; VLM badge = auto</span>'
        "</div>"
        '<div class="legend">'
        '<span class="text-muted">VLM badge:</span> '
        '<span class="badge badge-true">True</span> '
        '<span class="badge badge-false">False</span> '
        '<span class="badge badge-edited">Edited</span> '
        '<span class="badge badge-verified">Verified</span> '
        '<span class="badge badge-pending">Pending</span>'
        "</div>"
    )
    cross_tab_rows = []
    for gid, title in VISUAL_GROUP_ORDER[:4]:
        cross_tab_rows.append(
            f"<tr><th>{_esc(title)}</th><td>{visual_counts.get(gid, 0)}</td></tr>"
        )
    cross_tab = (
        '<table class="cross-tab">'
        "<caption>챗점 분류 (original × corr1)</caption>"
        "<thead><tr><th>Group</th><th>Count</th></tr></thead>"
        f"<tbody>{''.join(cross_tab_rows)}</tbody>"
        "</table>"
    )
    stats = (
        cross_tab
        + f"<p class=stats><b>챗점 groups</b> (n={n_total})</p>"
        f"<ul class=stats-list>"
        + "".join(
            f"<li>{_esc(title)}: {visual_counts.get(gid, 0)}</li>"
            for gid, title in VISUAL_GROUP_ORDER
        )
        + f"</ul>"
        f"<p class=stats><b>VLM badge</b> — "
        f"Gen True: {counts['gen_true']} · Gen False: {counts['gen_false']} · "
        f"Edited: {counts['corr_edited']} · Verified: {counts['corr_verified']} · "
        f"Pending: {counts['corr_pending']}"
        f"</p>"
    )
    single_cue = only_cue is not None
    cards_rule = (
        "grid-template-columns: 1fr;"
        if single_cue
        else "grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));"
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Correction overview — exp{exp_id} {model_tag}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f6f7fb; color: #1a1a1a; }}
h1 {{ margin-bottom: 8px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px; }}
.stats {{ color: #555; margin-bottom: 12px; }}
.stats-list {{ margin: 0 0 20px 1.2rem; color: #444; line-height: 1.6; }}
.cross-tab {{ border-collapse: collapse; font-size: 0.85rem; margin-bottom: 24px; }}
.cross-tab th, .cross-tab td {{ border: 1px solid #dde3ee; padding: 4px 12px; text-align: center; }}
.cross-tab caption {{ text-align: left; font-weight: 600; margin-bottom: 6px; color: #444; }}
.badge-row {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.badge-label {{ font-size: 0.72rem; color: #888; min-width: 2.8rem; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
.badge-true {{ background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }}
.badge-false {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
.badge-unknown {{ background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; }}
.badge-edited {{ background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }}
.badge-verified {{ background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; }}
.badge-pending {{ background: #e5e7eb; color: #374151; border: 1px solid #d1d5db; }}
.badge-other {{ background: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }}
.text-muted {{ color: #666; font-size: 0.82rem; }}
.group-section {{ margin-bottom: 36px; }}
.group-title {{ font-size: 1.15rem; margin: 0 0 14px; padding-bottom: 8px; border-bottom: 2px solid #dde3ee; }}
.group-count {{ color: #666; font-size: 0.95rem; font-weight: 500; }}
.cards {{ display: grid; {cards_rule} gap: 16px; }}
.card {{ background: #fff; border: 1px solid #dde3ee; border-radius: 10px; padding: 14px; }}
.head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; }}
.head h2 {{ margin: 0; font-size: 1rem; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.meta {{ font-size: 0.85rem; color: #555; margin: 8px 0 12px; }}
.media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
@media (max-width: 1200px) {{ .media-grid {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 720px) {{ .media-grid {{ grid-template-columns: 1fr; }} }}
figure {{ margin: 0; }}
figcaption {{ font-size: 0.8rem; color: #666; margin-bottom: 4px; }}
img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
.mp4 {{ font-size: 0.75rem; margin: 4px 0 0; }}
.cfg-toggle {{ margin-top: 8px; font-size: 0.8rem; }}
.cfg-toggle summary {{ cursor: pointer; color: #444; font-weight: 600; user-select: none; }}
.cfg-toggle summary:hover {{ color: #1e40af; }}
.cfg-toggle pre {{
  margin: 6px 0 0; padding: 8px; background: #f7f8fc; border: 1px solid #e8ecf4;
  border-radius: 6px; font-size: 0.72rem; line-height: 1.35; white-space: pre-wrap;
  word-break: break-word; max-height: 280px; overflow: auto;
}}
.cfg-missing {{ color: #999; font-size: 0.78rem; margin-top: 6px; }}
.vq {{ font-size: 0.82rem; margin-top: 10px; line-height: 1.4; }}
.missing {{ color: #c00; font-size: 0.85rem; }}
</style></head>
<body>
<h1>Correction overview (exp{exp_id}, {model_tag}) — new-render chain</h1>
{legend}
{stats}
{"".join(sections)}
</body></html>
"""
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--only", default=None, help="Single cue name")
    ap.add_argument("--out", type=Path, default=None, help="Output HTML path")
    args = ap.parse_args()
    path = build_overview_html(
        model_tag=args.model_tag,
        exp_id=args.exp,
        out_path=args.out,
        only_cue=args.only,
    )
    print(f"HTML saved: {path}")


if __name__ == "__main__":
    main()
