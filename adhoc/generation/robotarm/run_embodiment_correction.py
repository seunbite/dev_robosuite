#!/usr/bin/env python3
"""Correction (+recog-correct) for embodiment Motion Code renders (humanoid / roomba / …).

Runs two wrapper variants per job:
  - correct: standard Correction Q1/Q2 (known cue)
  - recog-correct: Recognition Q first (what does it look like?), then correct if ≠ cue

Example:
  source APIKEY.sh
  python adhoc/generation/robotarm/run_embodiment_correction.py \\
    --jobs 27:humanoid,28:humanoid,30:roomba
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402

Variant = Literal["correct", "recog-correct"]
Platform = Literal["humanoid", "roomba", "tiago", "quadruped", "single_arm"]

_WRAPPER_CORRECT = _REPO / "data/seed/prompt/_shared_correction_wrapper.txt"
_WRAPPER_RECOG = _REPO / "data/seed/prompt/_shared_recog_correction_wrapper.txt"
_CORRECTION_ONESHOT = (
    _REPO / "data/seed/shots/embodiment_correction/humanoid_depict_lion_oneshot.json"
)
_ALL_JSON = (
    _REPO / "data/results/motion_configs/embodiment_motion_code/embodiment_motion_code_all.json"
)
_RENDER_ROOT = _REPO / "data/results/render/embodiment_motion_code"
_OUT_ROOT = _REPO / "data/results/verify/embodiment_motion_code/correction"

# Full generation specs (prompt_exp0 level) — no short inline schemas.
_PROMPT_EXP0: dict[str, Path] = {
    "humanoid": _REPO / "data/seed/prompt/humanoid/exp/prompt_exp0.txt",
    "tiago": _REPO / "data/seed/prompt/tiago/exp/prompt_exp0.txt",
    "roomba": _REPO / "data/seed/prompt/roomba/exp/prompt_exp0.txt",
    "quadruped": _REPO / "data/seed/prompt/quadruped/exp/prompt_exp0.txt",
    "single_arm": _REPO / "data/seed/prompt/single_arm/exp/prompt_exp0.txt",
}

_RECOG_Q_RE = re.compile(
    r"^\s*#\s*Recognition\s*Q\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_CORR_Q1_RE = re.compile(
    r"^\s*#\s*Correction\s*Q1\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_CORR_Q2_RE = re.compile(
    r"^\s*#\s*Correction\s*Q2\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_MOTION_CODE_FENCE_RE = re.compile(
    r"```(?:motion[_\s-]?code|mc|text)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_jobs(spec: str, all_data: list[dict[str, Any]]) -> list[tuple[dict, Platform]]:
    by_idx = {int(e["idx"]): e for e in all_data}
    jobs: list[tuple[dict, Platform]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx_s, plat = part.split(":", 1)
        idx = int(idx_s.strip().lstrip("cC"))
        platform = plat.strip().lower()  # type: ignore[assignment]
        if idx not in by_idx:
            raise SystemExit(f"unknown idx {idx}")
        if platform not in by_idx[idx]["platforms"]:
            raise SystemExit(f"idx {idx} has no platform {platform}")
        jobs.append((by_idx[idx], platform))  # type: ignore[arg-type]
    return jobs


def _gif_path(idx: int, cue: str, platform: str) -> Path:
    return _RENDER_ROOT / platform / "gif" / f"emb_g{idx:02d}_{cue}.gif"


def _fill_placeholders(template: str, *, cue: str, description: str) -> str:
    """Fill {{CUE_NAME}} / {{CUE}} / {{DESCRIPTION}} / catalog / few-shot slots."""
    return (
        template.replace("{{CUE_NAME}}", cue)
        .replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{CUE_CATALOG}}", "")
        .replace("{{FEW_SHOT_EXAMPLES}}", "")
    )


def _inner_prompt(platform: Platform, cue: str, description: str) -> str:
    """Load full prompt_exp0 for this platform (no short inline schemas)."""
    path = _PROMPT_EXP0.get(platform)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"missing prompt_exp0 for platform={platform!r}: expected {path}"
        )
    return _fill_placeholders(path.read_text(encoding="utf-8"), cue=cue, description=description)


def _prior_attempts_block(row: dict[str, Any]) -> str:
    prior = {
        "cue": row.get("cue"),
        "description": row.get("description"),
        "source_code": row.get("source_code"),
        "movements": row.get("movements"),
    }
    return (
        "attempt 0 (current render — see attached GIF):\n"
        + json.dumps(prior, indent=2, ensure_ascii=False)
    )


def _correction_oneshot_block() -> str:
    """Format the shared humanoid correction one-shot for any-platform prompts."""
    if not _CORRECTION_ONESHOT.is_file():
        return ""
    shot = json.loads(_CORRECTION_ONESHOT.read_text(encoding="utf-8"))
    origin = shot.get("origin") or {}
    corrected = shot.get("corrected") or {}
    return (
        "\n[Correction one-shot example — **humanoid** "
        "(same discipline for every robot platform)]\n"
        f"Platform note: {shot.get('platform_note') or 'humanoid example; reuse pattern on other robots.'}\n"
        f"Cue: {shot.get('cue')}\n"
        f"Lesson: {shot.get('lesson') or ''}\n\n"
        "Origin attempt (failed):\n"
        f"Motion Code: {origin.get('source_code') or ''}\n"
        f"Config:\n{json.dumps(origin.get('movements') or [], indent=2, ensure_ascii=False)}\n\n"
        f"# Recognition Q: {shot.get('recognition_q') or ''}\n"
        f"# Correction Q1: {shot.get('correction_q1') or ''}\n"
        f"# Correction Q2: {shot.get('correction_q2') or ''}\n\n"
        "Corrected attempt (implements Q2 with bold parametric margins — copy this discipline):\n"
        f"Motion Code: {corrected.get('source_code') or ''}\n"
        f"Config:\n{json.dumps(corrected.get('movements') or [], indent=2, ensure_ascii=False)}\n"
    )


def _build_prompt(
    variant: Variant,
    *,
    cue: str,
    description: str,
    inner: str,
    row: dict[str, Any],
) -> str:
    path = _WRAPPER_RECOG if variant == "recog-correct" else _WRAPPER_CORRECT
    wrapper = path.read_text(encoding="utf-8")
    return (
        wrapper.replace("{{MEDIA_KIND}}", "animated GIF of the full rendered motion")
        .replace("{{CUMULATIVE_MEDIA_NOTE}}", "")
        .replace("{{CORRECTION_ONESHOT}}", _correction_oneshot_block())
        .replace("{{INNER_PROMPT}}", inner.strip())
        .replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{PRIOR_ATTEMPTS_BLOCK}}", _prior_attempts_block(row))
    )


def _parse_q(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text or ""))
    for m in reversed(matches):
        ans = m.group(1).strip()
        if not _is_prompt_echo(ans):
            return ans
    return None


def _is_prompt_echo(ans: str) -> bool:
    low = ans.lower().strip()
    if not low:
        return True
    if low.startswith("<") and ">" in low[:80]:
        return True
    if "without using the target cue name" in low:
        return True
    if "does the render look like the target cue" in low and "start with" in low:
        return True
    if "is the prior config" in low and "appropriate" in low:
        return True
    if low.startswith("if correction q1 is no"):
        return True
    if "how should" in low and ("motion" in low or "movements" in low) and "n/a" in low:
        return True
    if "do not echo" in low:
        return True
    if re.search(r'start with\s+[\"\']?yes', low):
        return True
    if "answer starting with" in low:
        return True
    return False


def _is_inner_planning_line(body: str) -> bool:
    """True for prompt_exp0 planning headings (# Q1..Q4), not Correction answers."""
    low = body.lower().strip()
    if re.match(r"^q[1-4]\s*:", low):
        return True
    if low.startswith("how would a human") or low.startswith("how would a wheeled"):
        return True
    if low.startswith("how would a four-legged"):
        return True
    if "candidates comparison" in low or "structure comparison" in low:
        return True
    if "context reinforcement" in low:
        return True
    return False


def _parse_recog_block(raw: str) -> tuple[str | None, str | None, str | None]:
    """Parse Recognition/Q1/Q2; tolerate echoed questions followed by bare answers."""
    recognition = _parse_q(_RECOG_Q_RE, raw)
    q1 = _parse_q(_CORR_Q1_RE, raw)
    q2 = _parse_q(_CORR_Q2_RE, raw)
    if recognition and q1 and q2:
        return recognition, q1, q2

    # Fallback: only consume answers tied to Recognition/Correction labels.
    # Never treat inner prompt_exp0 `# Q1:`…`# Q4:` planning lines as Correction answers.
    answers: list[str] = []
    awaiting_bare = False
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("```") or s.startswith("{"):
            if answers:
                break
            if s.startswith("{"):
                break
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if _is_inner_planning_line(body):
                awaiting_bare = False
                continue
            if re.match(r"^(Recognition\s*Q|Correction\s*Q[12])\s*:", body, re.I):
                labeled = body.split(":", 1)[1].strip() if ":" in body else ""
                if labeled and not _is_prompt_echo(labeled):
                    answers.append(labeled)
                    awaiting_bare = False
                else:
                    awaiting_bare = True
                continue
            if _is_prompt_echo(body):
                awaiting_bare = True
                continue
            # `# No, ...` style answer after a Correction echo
            if awaiting_bare and body and not _is_prompt_echo(body):
                answers.append(body)
                awaiting_bare = False
                if len(answers) >= 3:
                    break
            continue
        if awaiting_bare and not _is_prompt_echo(s) and not s.startswith("Motion Code"):
            if re.match(r"^(pose|move|path|hold)\(", s):
                break
            answers.append(s)
            awaiting_bare = False
            if len(answers) >= 3:
                break

    has_recog_header = bool(_RECOG_Q_RE.search(raw or ""))
    if has_recog_header:
        if not recognition and len(answers) >= 1:
            recognition = answers[0]
        if not q1 and len(answers) >= 2:
            q1 = answers[1]
        if not q2 and len(answers) >= 3:
            q2 = answers[2]
    else:
        if not q1 and len(answers) >= 1:
            q1 = answers[0]
        if not q2 and len(answers) >= 2:
            q2 = answers[1]
        recognition = None
    return recognition, q1, q2


def _normalize_movements(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix common VLM schema slips (flat joints / path fields)."""
    out: list[dict[str, Any]] = []
    for step in movements or []:
        if not isinstance(step, dict):
            continue
        step = dict(step)
        params = dict(step.get("parameters") or {})
        stype = str(step.get("type") or "").lower()
        if stype == "movement":
            if "movement" not in params:
                mv: dict[str, Any] = {}
                for k in ("joints", "speed", "hold_time", "repetition", "duration"):
                    if k in params:
                        mv[k] = params.pop(k)
                if "joints" in step:
                    mv.setdefault("joints", step.pop("joints"))
                if mv:
                    params["movement"] = mv
            mv = params.get("movement")
            if isinstance(mv, dict):
                for j in mv.get("joints") or []:
                    if not isinstance(j, dict):
                        continue
                    deg = j.get("degrees")
                    if isinstance(deg, (int, float)):
                        j["degrees"] = [0.0, float(deg)]
                    elif isinstance(deg, list) and len(deg) == 1 and isinstance(deg[0], (int, float)):
                        j["degrees"] = [0.0, float(deg[0])]
                    elif isinstance(deg, list) and len(deg) >= 2:
                        j["degrees"] = [float(deg[0]), float(deg[-1])]
        elif stype == "path":
            if "path" not in params:
                path: dict[str, Any] = {}
                for k in (
                    "mode",
                    "shape",
                    "axis",
                    "distance",
                    "plane",
                    "radius",
                    "sweep",
                    "degrees",
                    "x",
                    "y",
                    "speed",
                    "hold_time",
                    "repetition",
                    "side",
                    "direction",
                ):
                    if k in params:
                        path[k] = params.pop(k)
                if path:
                    params["path"] = path
        step["parameters"] = params
        out.append(step)
    return out


def _q1_yes(q1: str | None) -> bool | None:
    if not q1:
        return None
    low = q1.strip().lower()
    if re.match(r"^yes\b", low):
        return True
    if re.match(r"^no\b", low):
        return False
    return None


def _extract_motion_code(raw: str, fallback: str) -> str:
    m = _MOTION_CODE_FENCE_RE.search(raw or "")
    if m:
        return m.group(1).strip()
    # bare lines that look like motion code
    lines = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if re.match(r"^(pose|move|path|hold)\(", s):
            lines.append(s)
    if lines:
        # join with + if model emitted one call per line without +
        joined = "\n".join(lines)
        if "+" not in joined and len(lines) > 1:
            return "\n+ ".join(lines)
        return joined
    return fallback


def _extract_json(raw: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _call_gemini(*, prompt: str, gif: Path, model: str) -> str:
    from google import genai
    from google.genai import types

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
    client = genai.Client(api_key=key)
    parts: list[Any] = [
        types.Part.from_bytes(data=gif.read_bytes(), mime_type="image/gif"),
        prompt,
    ]
    resp = client.models.generate_content(model=model, contents=parts)
    return (resp.text or "").strip()


def _render_row(platform: Platform, row: dict[str, Any], out_gif: Path) -> Path:
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    scratch = out_gif.parent / f"_scratch_{out_gif.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        if platform == "humanoid":
            from embodiment_renderers import render_unitree_g1

            render_unitree_g1([row], scratch, force=True)
        elif platform == "roomba":
            from embodiment_renderers import render_roomba

            render_roomba([row], scratch, force=True)
        elif platform == "tiago":
            from motion_code_to_config import render_tiago

            # render_tiago writes into out_root/tiago/gif/
            render_tiago([row], out_root=scratch, force=True)
            produced = scratch / "tiago" / "gif" / f"emb_g{int(row['idx']):02d}_{row['cue']}.gif"
            if produced.is_file():
                out_gif.write_bytes(produced.read_bytes())
                mp4 = produced.with_suffix(".mp4")
                if mp4.is_file():
                    out_gif.with_suffix(".mp4").write_bytes(mp4.read_bytes())
                return out_gif
        else:
            raise NotImplementedError(platform)
        produced = scratch / f"emb_g{int(row['idx']):02d}_{row['cue']}.gif"
        if not produced.is_file():
            cands = list(scratch.rglob("*.gif"))
            if not cands:
                raise FileNotFoundError(f"no gif rendered for {row['cue']}")
            produced = cands[0]
        out_gif.write_bytes(produced.read_bytes())
        mp4 = produced.with_suffix(".mp4")
        if mp4.is_file():
            out_gif.with_suffix(".mp4").write_bytes(mp4.read_bytes())
        return out_gif
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _analyze_response(
    variant: Variant,
    *,
    platform: Platform,
    row: dict[str, Any],
    raw: str,
) -> dict[str, Any]:
    """Parse a VLM correction response into Q1/Q2, config, and action decision.

    Shared by round-1 (``run_one``) and later rounds so the parse/rewrite rules
    stay identical. ``row`` is the prior attempt (input) config.
    """
    recognition = None
    if variant == "recog-correct":
        recognition, q1, q2 = _parse_recog_block(raw)
    else:
        q1 = _parse_q(_CORR_Q1_RE, raw)
        q2 = _parse_q(_CORR_Q2_RE, raw)
        # also tolerate echo → bare answers for standard wrapper
        if not q1 or _is_prompt_echo(q1 or ""):
            answers: list[str] = []
            for line in (raw or "").splitlines():
                s = line.strip()
                if not s.startswith("#"):
                    continue
                body = s.lstrip("#").strip()
                if re.match(r"^Correction\s*Q[12]\s*:", body, re.I):
                    labeled = body.split(":", 1)[1].strip()
                    if labeled and not _is_prompt_echo(labeled):
                        answers.append(labeled)
                    continue
                if _is_prompt_echo(body):
                    continue
            if (not q1 or _is_prompt_echo(q1)) and answers:
                q1 = answers[0]
            if (not q2 or _is_prompt_echo(q2)) and len(answers) > 1:
                q2 = answers[1]
    is_ok = _q1_yes(q1)

    obj = _extract_json(raw) or {}
    source_code = str(obj.get("source_code") or "").strip() or _extract_motion_code(
        raw, str(row.get("source_code") or "")
    )
    converted = convert_code(source_code, platform=platform)
    movements = obj.get("movements") if isinstance(obj.get("movements"), list) else None
    if movements:
        movements = _normalize_movements(movements)
    else:
        movements = converted.get("movements") or row.get("movements")

    prior_moves = json.dumps(row.get("movements") or [], sort_keys=True)
    new_moves = json.dumps(movements or [], sort_keys=True)
    movements_changed = prior_moves != new_moves
    code_changed = source_code.strip() != str(row.get("source_code") or "").strip()
    # google prompt_exp0 often emits movements JSON only — keep that as the
    # authoritative artifact and avoid showing the stale prior Motion Code.
    primary_display = "motion_code"
    if movements_changed and not code_changed:
        primary_display = "config"
        source_code = str(row.get("source_code") or "")
    action = "keep"
    if is_ok is False or movements_changed or (is_ok is None and code_changed):
        action = "rewrite"
    # If model skipped Correction answers but still emitted a new config, treat as rewrite.
    if action == "keep" and is_ok is None and (movements_changed or code_changed):
        action = "rewrite"

    out_row = dict(row)
    out_row["movements"] = movements
    out_row["source_code"] = source_code
    out_row["primary_display"] = primary_display
    out_row["correction_variant"] = variant
    out_row["time"] = datetime.now().isoformat(timespec="seconds")

    return {
        "recognition": recognition,
        "q1": q1,
        "q2": q2,
        "is_ok": is_ok,
        "source_code": source_code,
        "movements": movements,
        "primary_display": primary_display,
        "action": action,
        "out_row": out_row,
    }


def run_one(
    entry: dict[str, Any],
    platform: Platform,
    variant: Variant,
    *,
    model: str,
    force: bool,
) -> dict[str, Any]:
    idx = int(entry["idx"])
    cue = str(entry["cue"])
    row = dict(entry["platforms"][platform])
    cue_dir = _OUT_ROOT / variant / platform / f"g{idx:02d}_{cue}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    record_path = cue_dir / "correction.json"
    out_gif = cue_dir / f"{variant}_{cue}.gif"

    if record_path.is_file() and out_gif.is_file() and not force:
        print(f"[skip] {variant} {platform} {cue}", flush=True)
        return json.loads(record_path.read_text(encoding="utf-8"))

    gif = _gif_path(idx, cue, platform)
    if not gif.is_file():
        raise FileNotFoundError(gif)

    description = str(row.get("description") or f"Motion-code embodiment ({platform}) for {cue}")
    inner = _inner_prompt(platform, cue, description)
    prompt = _build_prompt(variant, cue=cue, description=description, inner=inner, row=row)
    (cue_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    print(f"[vlm] {variant} {platform} {cue} …", flush=True)
    raw = _call_gemini(prompt=prompt, gif=gif, model=model)
    (cue_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

    analysis = _analyze_response(variant, platform=platform, row=row, raw=raw)
    recognition = analysis["recognition"]
    q1 = analysis["q1"]
    q2 = analysis["q2"]
    is_ok = analysis["is_ok"]
    source_code = analysis["source_code"]
    movements = analysis["movements"]
    primary_display = analysis["primary_display"]
    action = analysis["action"]
    out_row = analysis["out_row"]

    print(f"[render] {variant} {platform} {cue} ({action})", flush=True)
    if action == "rewrite" or not out_gif.is_file() or force:
        _render_row(platform, out_row, out_gif)
    else:
        # still copy original for side-by-side if kept
        out_gif.write_bytes(gif.read_bytes())

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "platform": platform,
        "idx": idx,
        "cue": cue,
        "model": model,
        "recognition_q": recognition,
        "correction_q1": q1,
        "correction_q2": q2,
        "is_correct": is_ok,
        "action": action,
        "primary_display": primary_display,
        "input_source_code": row.get("source_code"),
        "output_source_code": source_code if primary_display == "motion_code" else "",
        "output_row": out_row,
        "media": {"prior_gif": str(gif), "corrected_gif": str(out_gif)},
        "raw_response_path": str(cue_dir / "raw_response.txt"),
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (cue_dir / "corrected_config.json").write_text(
        json.dumps(out_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def run_next_round(
    prev_record: dict[str, Any],
    *,
    model: str,
    force: bool,
) -> dict[str, Any]:
    """Run one more correction round on top of an existing correction record.

    The prior round's corrected GIF becomes the media input and its corrected
    config becomes the prior attempt (``correct`` → corr-corr, ``recog-correct``
    → reccorr-reccorr). Output lands in a ``round{N}`` subfolder of the same cue.
    """
    variant: Variant = prev_record["variant"]  # type: ignore[assignment]
    platform: Platform = prev_record["platform"]  # type: ignore[assignment]
    idx = int(prev_record["idx"])
    cue = str(prev_record["cue"])
    prev_round = int(prev_record.get("round", 1))
    this_round = prev_round + 1

    row = dict(prev_record.get("output_row") or {})
    row.setdefault("cue", cue)
    prior_gif = Path(prev_record["media"]["corrected_gif"])
    if not prior_gif.is_file():
        raise FileNotFoundError(prior_gif)

    cue_dir = _OUT_ROOT / variant / platform / f"g{idx:02d}_{cue}" / f"round{this_round}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    record_path = cue_dir / "correction.json"
    out_gif = cue_dir / f"{variant}_r{this_round}_{cue}.gif"

    if record_path.is_file() and out_gif.is_file() and not force:
        print(f"[skip] r{this_round} {variant} {platform} {cue}", flush=True)
        return json.loads(record_path.read_text(encoding="utf-8"))

    description = str(row.get("description") or f"Motion-code embodiment ({platform}) for {cue}")
    inner = _inner_prompt(platform, cue, description)
    prompt = _build_prompt(variant, cue=cue, description=description, inner=inner, row=row)
    (cue_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    print(f"[vlm] r{this_round} {variant} {platform} {cue} …", flush=True)
    raw = _call_gemini(prompt=prompt, gif=prior_gif, model=model)
    (cue_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

    analysis = _analyze_response(variant, platform=platform, row=row, raw=raw)
    action = analysis["action"]
    out_row = analysis["out_row"]
    source_code = analysis["source_code"]
    primary_display = analysis["primary_display"]

    print(f"[render] r{this_round} {variant} {platform} {cue} ({action})", flush=True)
    if action == "rewrite" or not out_gif.is_file() or force:
        _render_row(platform, out_row, out_gif)
    else:
        out_gif.write_bytes(prior_gif.read_bytes())

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "platform": platform,
        "idx": idx,
        "cue": cue,
        "round": this_round,
        "model": model,
        "recognition_q": analysis["recognition"],
        "correction_q1": analysis["q1"],
        "correction_q2": analysis["q2"],
        "is_correct": analysis["is_ok"],
        "action": action,
        "primary_display": primary_display,
        "input_source_code": row.get("source_code"),
        "output_source_code": source_code if primary_display == "motion_code" else "",
        "output_row": out_row,
        "media": {"prior_gif": str(prior_gif), "corrected_gif": str(out_gif)},
        "raw_response_path": str(cue_dir / "raw_response.txt"),
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (cue_dir / "corrected_config.json").write_text(
        json.dumps(out_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def _code_config_blocks(
    *,
    source_code: str | None,
    movements: list[dict[str, Any]] | None,
) -> str:
    code = html_mod.escape((source_code or "").strip() or "(empty)")
    cfg = html_mod.escape(json.dumps(movements or [], indent=2, ensure_ascii=False))
    return (
        f"<div class='mc'><div class='mc-label'>Config</div><pre class='mc'>{cfg}</pre></div>"
        f"<div class='mc'><div class='mc-label'>Motion Code</div><pre class='mc'>{code}</pre></div>"
    )


def build_html(records: list[dict[str, Any]], out_html: Path) -> Path:
    # Load current (or pre-correction) platform rows for the original column.
    all_data = json.loads(_ALL_JSON.read_text(encoding="utf-8"))
    by_idx = {int(e["idx"]): e for e in all_data}

    # Index any later-round records (round2+) by (variant, platform, idx, round).
    later_rounds: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for p in _OUT_ROOT.glob("*/*/*/round*/correction.json"):
        try:
            rr = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (str(rr.get("variant")), str(rr.get("platform")), int(rr["idx"]))
        later_rounds.setdefault(key, []).append(rr)
    for lst in later_rounds.values():
        lst.sort(key=lambda r: int(r.get("round", 2)))

    sections: list[str] = []
    # group by (idx, platform)
    keys: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for r in records:
        k = (int(r["idx"]), str(r["platform"]))
        if k not in seen:
            seen.add(k)
            keys.append((k[0], k[1], str(r["cue"])))

    for idx, platform, cue in keys:
        cols = []
        # original — show both config + motion code
        orig = _gif_path(idx, cue, platform)
        orig_rel = os.path.relpath(orig, out_html.parent) if orig.is_file() else ""
        orig_row = ((by_idx.get(idx) or {}).get("platforms") or {}).get(platform) or {}
        # Prefer the correction input snapshot when available (true pre-correction original).
        sample = next(
            (x for x in records if int(x["idx"]) == idx and x["platform"] == platform),
            None,
        )
        orig_code = str((sample or {}).get("input_source_code") or orig_row.get("source_code") or "")
        # Reconstruct original movements from input if output_row is corrected-only:
        # use all.json only when it still matches input code; otherwise keep input code
        # and movements from a sidecar if present. Fall back to all.json movements.
        orig_moves = orig_row.get("movements") or []
        if sample and str(orig_row.get("source_code") or "").strip() != str(sample.get("input_source_code") or "").strip():
            # all.json already promoted to corrected — recover original movements from
            # convert_code(input) when possible.
            try:
                orig_moves = convert_code(orig_code, platform=platform).get("movements") or orig_moves
            except Exception:
                pass
        cols.append(
            "<div class='col'><h3>original</h3>"
            + (f"<img src='{html_mod.escape(orig_rel)}' loading='lazy'>" if orig_rel else "<p class='miss'>missing</p>")
            + _code_config_blocks(source_code=orig_code, movements=orig_moves)
            + "</div>"
        )
        for variant in ("correct", "recog-correct"):
            rec = next(
                (x for x in records if int(x["idx"]) == idx and x["platform"] == platform and x["variant"] == variant),
                None,
            )
            if not rec:
                cols.append(f"<div class='col'><h3>{variant}</h3><p class='miss'>n/a</p></div>")
                continue
            gif = Path(rec["media"]["corrected_gif"])
            rel = os.path.relpath(gif, out_html.parent) if gif.is_file() else ""
            recog = html_mod.escape(str(rec.get("recognition_q") or "—"))
            q1 = html_mod.escape(str(rec.get("correction_q1") or ""))
            q2 = html_mod.escape(str(rec.get("correction_q2") or ""))
            action = html_mod.escape(str(rec.get("action") or ""))
            out_row = rec.get("output_row") or {}
            movements = out_row.get("movements") or []
            out_code = str(rec.get("output_source_code") or out_row.get("source_code") or "").strip()
            cols.append(
                f"<div class='col'><h3>{variant} <span class='tag'>{action}</span></h3>"
                + (f"<img src='{html_mod.escape(rel)}' loading='lazy'>" if rel else "<p class='miss'>no gif</p>")
                + f"<div class='meta'><b>Recognition:</b> {recog}<br>"
                f"<b>Q1:</b> {q1}<br><b>Q2:</b> {q2}</div>"
                + _code_config_blocks(source_code=out_code, movements=movements)
                + "</div>"
            )
            # Extra columns for later correction rounds (corr-corr / reccorr-reccorr).
            for rr in later_rounds.get((variant, platform, idx), []):
                rnd = int(rr.get("round", 2))
                rgif = Path(rr["media"]["corrected_gif"])
                rrel = os.path.relpath(rgif, out_html.parent) if rgif.is_file() else ""
                rrecog = html_mod.escape(str(rr.get("recognition_q") or "—"))
                rq1 = html_mod.escape(str(rr.get("correction_q1") or ""))
                rq2 = html_mod.escape(str(rr.get("correction_q2") or ""))
                raction = html_mod.escape(str(rr.get("action") or ""))
                r_out_row = rr.get("output_row") or {}
                r_moves = r_out_row.get("movements") or []
                r_code = str(rr.get("output_source_code") or r_out_row.get("source_code") or "").strip()
                label = f"{variant} →r{rnd}"
                cols.append(
                    f"<div class='col'><h3>{html_mod.escape(label)} <span class='tag'>{raction}</span></h3>"
                    + (f"<img src='{html_mod.escape(rrel)}' loading='lazy'>" if rrel else "<p class='miss'>no gif</p>")
                    + f"<div class='meta'><b>Recognition:</b> {rrecog}<br>"
                    f"<b>Q1:</b> {rq1}<br><b>Q2:</b> {rq2}</div>"
                    + _code_config_blocks(source_code=r_code, movements=r_moves)
                    + "</div>"
                )
        sections.append(
            f"<section class='cue'><h2>c{idx:02d} {html_mod.escape(cue)} "
            f"<span class='tag'>{html_mod.escape(platform)}</span></h2>"
            f"<div class='grid'>{''.join(cols)}</div></section>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Embodiment correction (correct vs recog-correct)</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta-top{{padding:10px 24px;color:#9aa0a6;font-size:0.9rem}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.col{{background:#12151c;border:1px solid #2a2f3a;border-radius:8px;padding:10px}}
.col img{{width:100%;height:auto;border-radius:6px;background:#000}}
.tag{{font-size:0.75rem;background:#2a3140;padding:2px 8px;border-radius:999px;margin-left:6px}}
.meta{{font-size:0.82rem;color:#c5cad3;margin-top:8px;line-height:1.35}}
.mc-label{{font-size:0.68rem;color:#9aa0a6;margin:8px 0 3px;text-transform:uppercase;letter-spacing:0.04em}}
.mc{{white-space:pre-wrap;font-size:0.75rem;background:#0b0d12;padding:8px;border-radius:6px;max-height:180px;overflow:auto}}
.miss{{color:#f28b82}}
</style></head><body>
<h1>Embodiment correction — correct vs recog-correct</h1>
<p class="meta-top">{html_mod.escape(datetime.now().isoformat(timespec="seconds"))} · {len(keys)} jobs</p>
{"".join(sections)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jobs",
        default="27:humanoid,28:humanoid,30:roomba",
        help="Comma list of idx:platform (e.g. 27:humanoid,30:roomba)",
    )
    ap.add_argument(
        "--variants",
        default="correct,recog-correct",
        help="Comma list: correct,recog-correct",
    )
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--round2",
        action="store_true",
        help="Run one more correction round on every existing round-1 record "
        "(correct→corr-corr, recog-correct→reccorr-reccorr).",
    )
    ap.add_argument(
        "--html",
        type=Path,
        default=_REPO / "data/results/html/embodiment_motion_code_correction.html",
    )
    args = ap.parse_args()

    if args.round2:
        prev = []
        for p in sorted(_OUT_ROOT.glob("*/*/*/correction.json")):
            try:
                prev.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        prev.sort(key=lambda r: (int(r["idx"]), str(r["platform"]), str(r["variant"])))
        for rec in prev:
            run_next_round(rec, model=args.model, force=args.force)
        # Rebuild HTML from round-1 records; build_html auto-attaches later rounds.
        merged = list(prev)
        merged.sort(key=lambda r: (int(r["idx"]), str(r["platform"]), str(r["variant"])))
        html_path = build_html(merged, args.html)
        print(f"HTML: {html_path}", flush=True)
        return

    all_data = json.loads(_ALL_JSON.read_text(encoding="utf-8"))
    jobs = _load_jobs(args.jobs, all_data)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]  # type: ignore[misc]

    records: list[dict[str, Any]] = []
    for entry, platform in jobs:
        for variant in variants:
            if variant not in ("correct", "recog-correct"):
                raise SystemExit(f"bad variant {variant}")
            rec = run_one(entry, platform, variant, model=args.model, force=args.force)  # type: ignore[arg-type]
            records.append(rec)

    # Merge with any prior correction records so the HTML stays cumulative.
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for p in _OUT_ROOT.glob("*/*/*/correction.json"):
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_key[(str(old.get("variant")), str(old.get("platform")), int(old["idx"]))] = old
    for rec in records:
        by_key[(str(rec.get("variant")), str(rec.get("platform")), int(rec["idx"]))] = rec
    merged = list(by_key.values())
    merged.sort(key=lambda r: (int(r["idx"]), str(r["platform"]), str(r["variant"])))

    html_path = build_html(merged, args.html)
    print(f"HTML: {html_path}", flush=True)


if __name__ == "__main__":
    main()
