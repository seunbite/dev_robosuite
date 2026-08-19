"""Build wrapped correction prompts (inner exp1/exp7 + outer Correction Q1/Q2) and parse responses."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_GOOGLE = _REPO / "adhoc/generation/tiago"
import sys

for p in (_REPO, _HERE, _GOOGLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pilot90_paths import SHOTS as MANIP_SHOTS, prompt_exp_path as manip_prompt_exp_path  # noqa: E402

ExpId = Literal["1", "7", "15"]
InnerExpId = Literal["0", "1", "7", "15"]
Domain = Literal["manipulator", "tiago"]

_WRAPPER_PATH = _REPO / "data/seed/prompt/_shared_correction_wrapper.txt"

_CORRECTION_Q1_RE = re.compile(
    r"^\s*#\s*Correction\s*Q1\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_CORRECTION_Q2_RE = re.compile(
    r"^\s*#\s*Correction\s*Q2\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _load_wrapper_template() -> str:
    return _WRAPPER_PATH.read_text(encoding="utf-8")


def _media_kind(domain: Domain, exp_id: ExpId) -> str:
    if exp_id in ("7", "15"):
        return "video (full rendered motion MP4)"
    return "animated GIF of the full rendered motion sequence"


def build_inner_prompt_manipulator_exp15(*, cue: str) -> str:
    template = manip_prompt_exp_path(15).read_text(encoding="utf-8")
    return template.replace("{{FEW_SHOT_EXAMPLES}}", "").replace("{{CUE_NAME}}", cue)


def build_inner_prompt_google_exp15(*, cue: str) -> str:
    template = (_REPO / "data/seed/prompt/tiago/exp/prompt_exp15.txt").read_text(encoding="utf-8")
    return template.replace("{{FEW_SHOT_EXAMPLES}}", "").replace("{{CUE_NAME}}", cue).replace(
        "{{CUE}}", cue
    )


def build_inner_prompt_manipulator(
    exp_id: ExpId,
    *,
    cue: str,
    row: dict[str, Any],
    description: str,
) -> str:
    from config_gen_vlm import _fewshot_block, _legacy_core  # noqa: WPS433

    if exp_id == "1":
        ppath = manip_prompt_exp_path(1)
        template = ppath.read_text(encoding="utf-8")
        examples = _fewshot_block(ppath, MANIP_SHOTS)
        return template.replace("{{FEW_SHOT_EXAMPLES}}", examples).replace("{{CUE_NAME}}", cue)

    fixed = row.get("gt_fixed_first_pose") or _first_pose_dict(row)
    ppath = manip_prompt_exp_path(7)
    template = ppath.read_text(encoding="utf-8")
    _, _format_example_block, _load_json_list, _, _resolve_shots_json, *_ = _legacy_core()
    shots = _load_json_list(_resolve_shots_json(str(manip_prompt_exp_path(1)), str(MANIP_SHOTS)))
    examples_parts: list[str] = []
    for ex in [c for c in shots if c.get("state") in ("handmade", "choreography")][:6]:
        mov = ex.get("movements", [])
        if mov and mov[0].get("type") == "pose":
            examples_parts.append(
                _format_example_block(
                    {k: v for k, v in ex.items() if k != "planning_shot"},
                    "# EXAMPLE (start pose + follow-on steps):",
                )
            )
    examples = "\n\n".join(examples_parts)
    return (
        template.replace("{{FEW_SHOT_EXAMPLES}}", examples)
        .replace("{{CUE_NAME}}", cue)
        .replace("{{FIXED_FIRST_POSE_JSON}}", json.dumps(fixed, indent=2, ensure_ascii=False))
    )


def build_inner_prompt_google(
    exp_id: InnerExpId,
    *,
    cue: str,
    row: dict[str, Any],
    description: str,
    backend: str = "gemini",
) -> str:
    from config_gen_mobile_vlm import (  # noqa: WPS433
        _build_exp0_prompt,
        _build_exp1_prompt,
        _build_exp7_prompt,
        tiago_gt_fixed_pose,
    )
    from pilot40_paths import load_gt_by_cue  # noqa: WPS433

    if exp_id == "0":
        return _build_exp0_prompt(cue=cue, backend=backend)
    if exp_id == "1":
        return _build_exp1_prompt(cue=cue, backend=backend)

    fixed = row.get("gt_fixed_first_pose")
    if not fixed:
        gt_row = load_gt_by_cue().get(cue) or {}
        fixed = tiago_gt_fixed_pose(gt_row)
    return _build_exp7_prompt(
        cue=cue,
        description=description,
        fixed_pose=fixed,
        backend=backend,
    )


def build_inner_prompt(
    domain: Domain,
    inner_exp_id: InnerExpId,
    *,
    cue: str,
    row: dict[str, Any],
    description: str,
    backend: str = "gemini",
) -> str:
    if domain == "manipulator":
        if inner_exp_id == "15":
            return build_inner_prompt_manipulator_exp15(cue=cue)
        if inner_exp_id == "0":
            raise ValueError("exp0 inner prompt is tiago only; use --inner-exp 1 or 7")
        return build_inner_prompt_manipulator(inner_exp_id, cue=cue, row=row, description=description)  # type: ignore[arg-type]
    if inner_exp_id == "15":
        return build_inner_prompt_google_exp15(cue=cue)
    return build_inner_prompt_google(
        inner_exp_id, cue=cue, row=row, description=description, backend=backend
    )


def _config_summary_json(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "cue": row.get("cue"),
            "description": row.get("description"),
            "movements": row.get("movements"),
        },
        indent=2,
        ensure_ascii=False,
    )


def format_prior_attempts_block(attempts: list[tuple[str, dict[str, Any]]]) -> str:
    """Format labeled prior configs for cumulative correction prompts."""
    parts: list[str] = []
    for label, row in attempts:
        parts.append(f"{label}\nconfig JSON:\n{_config_summary_json(row)}")
    return "\n\n".join(parts)


def cumulative_media_sentence(n_tiles: int) -> str:
    """One-line guide: tile order matches config block order below."""
    if n_tiles <= 1:
        return ""
    labels = ["original exp1 generation"] + [f"correction {i}" for i in range(1, n_tiles)]
    ordered = ", ".join(f"({i}) {labels[i - 1]}" for i in range(1, n_tiles + 1))
    return (
        f"The attached GIF is a single tiled clip: tiles run left-to-right as {ordered}, "
        f"and each tile matches the config block below in the same order."
    )


def build_wrapped_correction_prompt(
    domain: Domain,
    exp_id: ExpId,
    *,
    cue: str,
    row: dict[str, Any],
    description: str,
    backend: str = "gemini",
    inner_exp_id: InnerExpId | None = None,
    prior_attempts: list[tuple[str, dict[str, Any]]] | None = None,
) -> str:
    inner_id: InnerExpId = inner_exp_id if inner_exp_id is not None else exp_id  # type: ignore[assignment]
    inner = build_inner_prompt(
        domain, inner_id, cue=cue, row=row, description=description, backend=backend
    )
    attempts = prior_attempts if prior_attempts else [("Prior attempt (latest)", row)]
    n_tiles = len(attempts)
    if n_tiles > 1 and domain == "tiago":
        media_kind = (
            f"animated GIF with {n_tiles} tiles (left-to-right; one full motion per tile)"
        )
    else:
        media_kind = _media_kind(domain, exp_id)
    cumulative_note = cumulative_media_sentence(n_tiles) if n_tiles > 1 else ""
    if cumulative_note:
        cumulative_note = "\n" + cumulative_note
    wrapper = _load_wrapper_template()
    return (
        wrapper.replace("{{MEDIA_KIND}}", media_kind)
        .replace("{{CUMULATIVE_MEDIA_NOTE}}", cumulative_note)
        .replace("{{INNER_PROMPT}}", inner.strip())
        .replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{PRIOR_ATTEMPTS_BLOCK}}", format_prior_attempts_block(attempts))
    )


def _first_pose_dict(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return row.get("gt_fixed_first_pose") or {}


def _parse_correction_q1_is_correct(q1_text: str | None) -> bool | None:
    if not q1_text:
        return None
    low = q1_text.strip().lower()
    if re.match(r"^yes\b", low) or low.startswith("yes —") or low.startswith("yes -"):
        return True
    if re.match(r"^no\b", low) or low.startswith("no —") or low.startswith("no -"):
        return False
    if "not appropriate" in low or "inappropriate" in low:
        return False
    if "appropriate" in low and "not " not in low[:30]:
        return True
    return None


def _extract_reasoning_and_json_local(raw_text: str) -> tuple[str, dict[str, Any]]:
    """Lightweight JSON extract (avoids importing config_gen_single / mujoco chain)."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    reasoning = text[: m.start()].strip()
    return reasoning, obj


def _is_correction_q_prompt_echo(q_num: int, ans: str) -> bool:
    """True when the model echoed the wrapper question instead of answering."""
    low = ans.lower().strip()
    if not low:
        return True
    if q_num == 1:
        if "is the prior config" in low and "appropriate" in low:
            return True
        if re.search(r"start with\s+[\"']?yes", low):
            return True
    elif q_num == 2:
        if low.startswith("if correction q1 is no"):
            return True
        if "how should" in low and ("movements" in low or "entire" in low):
            return True
        if "cover pose" in low and "path/movement" in low:
            return True
        if re.search(r"if yes,\s*write\s+[\"']?n/a", low):
            return True
    return False


def _extract_correction_q_line(raw_text: str, q_num: int) -> str | None:
    """Parse ``# Correction Qn:`` answer; skip wrapper echo lines."""
    lines = (raw_text or "").splitlines()
    label = rf"^\s*#\s*Correction\s*Q{q_num}\s*:\s*(.*)$"
    candidates: list[str] = []
    prompt_pat = re.compile(label, re.IGNORECASE)
    for i, line in enumerate(lines):
        m = prompt_pat.match(line)
        if not m:
            continue
        ans = m.group(1).strip()
        if _is_correction_q_prompt_echo(q_num, ans):
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt.startswith("#"):
                    continue
                alt = nxt.lstrip("#").strip()
                if alt and not _is_correction_q_prompt_echo(q_num, alt):
                    candidates.append(alt)
                    break
            continue
        if ans:
            candidates.append(ans)
    return candidates[0] if candidates else None


def correction_q2_is_na(q2_text: str | None) -> bool:
    if not q2_text:
        return False
    low = q2_text.strip().lower()
    return low in ("n/a", "na", "n.a.", "none") or low.startswith("n/a")


def correction_indicates_verified_ok(correction: dict[str, Any]) -> bool:
    """True when Correction Q1=yes (including Q2=n/a shorthand)."""
    if correction.get("is_correct") is True:
        return True
    raw = correction.get("raw_text") or correction.get("inner_reasoning") or ""
    q1 = correction.get("correction_q1") or _extract_correction_q_line(raw, 1)
    q2 = correction.get("correction_q2") or _extract_correction_q_line(raw, 2)
    if correction_q2_is_na(q2):
        return True
    return _parse_correction_q1_is_correct(q1) is True


def parse_correction_response(
    raw_text: str,
    *,
    exp_id: ExpId,
) -> dict[str, Any]:
    """Parse Correction Q1/Q2 comments plus inner planning + config JSON."""
    correction_q1 = _extract_correction_q_line(raw_text, 1)
    correction_q2 = _extract_correction_q_line(raw_text, 2)
    is_correct = _parse_correction_q1_is_correct(correction_q1)
    if is_correct is not True and correction_q2_is_na(correction_q2):
        is_correct = True

    reasoning = ""
    config_parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        reasoning, config_parsed = _extract_reasoning_and_json_local(raw_text)
    except Exception as e:
        parse_error = str(e)
        try:
            config_parsed = json.loads(
                re.search(r"\{.*\}", raw_text, flags=re.S).group(0)  # type: ignore[union-attr]
            )
        except Exception:
            config_parsed = None

    return {
        "exp_id": exp_id,
        "correction_q1": correction_q1,
        "correction_q2": correction_q2,
        "is_correct": is_correct,
        "inner_reasoning": reasoning,
        "config_parsed": config_parsed,
        "parse_error": parse_error,
        "raw_text": raw_text,
    }


def _strip_leading_poses(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(movements)
    while out and out[0].get("type") == "pose":
        out.pop(0)
    return out


def row_from_parsed_config(
    domain: Domain,
    exp_id: ExpId,
    original: dict[str, Any],
    parsed: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Turn parsed correction JSON into a full config row."""
    cue = str(original.get("cue", ""))
    cue_idx = int(original.get("idx", 0))
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")

    if exp_id == "7":
        if domain == "manipulator":
            fixed = original.get("gt_fixed_first_pose") or _first_pose_dict(original)
            fixed_step = {
                "type": "pose",
                "parameters": {"pose": deepcopy(fixed), "speed": 1.0, "hold_time": 0.0},
            }
            tail = _strip_leading_poses(parsed.get("movements") or [])
            movements = [fixed_step, *tail]
            state = "exp7_motion_correction"
        else:
            from config_gen_mobile_vlm import _fixed_pose_step  # noqa: WPS433

            fixed = original.get("gt_fixed_first_pose") or _first_pose_dict(original)
            tail = _strip_leading_poses(parsed.get("movements") or [])
            movements = [_fixed_pose_step(fixed)] + tail
            state = "exp7_motion_correction"
    else:
        movements = parsed.get("movements") or []
        state = "exp1_pose_correction"
    if str(exp_id) == "15":
        state = "exp15_mime_correction"

    out = {
        "idx": cue_idx,
        "cue": cue,
        "description": parsed.get("description") or original.get("description", ""),
        "movements": movements,
        "state": state,
        "model": model,
        "time": now,
        "experiment": f"exp{exp_id}",
        "generation_valid": True,
    }
    if domain == "tiago":
        from adhoc.generation.tiago.mobile_pose_normalize import normalize_config_row  # noqa: WPS433

        out = normalize_config_row(out)
    if exp_id == "7":
        out["gt_fixed_first_pose"] = original.get("gt_fixed_first_pose") or _first_pose_dict(original)
        if original.get("pose_gt"):
            out["pose_gt"] = original["pose_gt"]
        if original.get("groundtruth"):
            out["groundtruth"] = original["groundtruth"]
    return out
