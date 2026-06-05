"""Experiment specs + accuracy extraction for pilot 32B suite."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]

# User backlog (7 settings), all Qwen2.5-VL-32B
EXPERIMENT_SPECS: list[dict[str, Any]] = [
    {
        "id": "1_multitile_grid6_pilot20",
        "title": "Multitile GT pick — grid 6, pilot 20",
        "kind": "multitile",
        "max_cues": 20,
        "grid_sizes": "6",
        "out_name": "exp1_multitile_grid6_pilot20.json",
    },
    {
        "id": "2_multitile_grid12_pilot20",
        "title": "Multitile GT pick — grid 12, pilot 20",
        "kind": "multitile",
        "max_cues": 20,
        "grid_sizes": "12",
        "out_name": "exp2_multitile_grid12_pilot20.json",
    },
    {
        "id": "3_pairwise_pilot20",
        "title": "Pairwise 2-way iconic pose, pilot 20",
        "kind": "pairwise",
        "max_cues": 20,
        "out_name": "exp3_pairwise_pilot20.json",
    },
    {
        "id": "4_fewshot_tile_pilot20",
        "title": "Few-shot single-tile verify, pilot 20",
        "kind": "fewshot",
        "max_cues": 20,
        "out_name": "exp4_fewshot_pilot20.json",
    },
    {
        "id": "5_temporal_multitile",
        "title": "Temporal cues — multitile grid 6+12 (tempo-aware prompt)",
        "kind": "multitile",
        "grid_sizes": "6,12",
        "cue_filter": "temporal",
        "temporal_prompt": True,
        "out_name": "exp5_temporal_multitile.json",
    },
    {
        "id": "6_google_robot_compare40",
        "title": "Google Robot pose compare (~40 cues)",
        "kind": "google_robot",
        "limit": 40,
        "out_name": "exp6_google_robot_compare40.json",
    },
    {
        "id": "7_multitile_pilot100",
        "title": "Multitile GT pick — grid 6+12, pilot 100",
        "kind": "multitile",
        "max_cues": 100,
        "grid_sizes": "6,12",
        "out_name": "exp7_multitile_pilot100.json",
    },
]

TEMPORAL_HASHTAGS = frozenset(
    {
        "rhythmic",
        "repetition",
        "oscillatory",
        "repetitive_pull",
        "repetitive",
        "come_here",
        "beckon",
        "tempo",
        "dynamic_temporal",
    }
)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or (_REPO / "data/seed/yml/pilot100_manifest.yml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _iter_manifest_cues(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, block in manifest.items():
        if not isinstance(block, dict) or "cues" not in block:
            continue
        for c in block["cues"]:
            if isinstance(c, dict) and c.get("cue"):
                out.append({**c, "_section": key})
    return out


def temporal_cue_names(manifest: dict[str, Any] | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    names: list[str] = []
    for c in _iter_manifest_cues(manifest):
        tags = set(c.get("hashtags") or [])
        if tags & TEMPORAL_HASHTAGS:
            names.append(str(c["cue"]))
    return sorted(set(names))


def pilot20_cue_names(consolidated_path: Path) -> list[str]:
    data = json.loads(consolidated_path.read_text(encoding="utf-8"))
    rows = sorted(data.get("rows") or [], key=lambda r: int(r.get("cue_idx", 0)))
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        cue = r.get("cue")
        if cue and cue not in seen:
            seen.add(cue)
            out.append(cue)
        if len(out) >= 20:
            break
    return out


def metrics_from_json(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}

    data = json.loads(path.read_text(encoding="utf-8"))
    kind = spec["kind"]

    if kind == "multitile":
        summary = data.get("summary") or {}
        rows_out: list[dict[str, Any]] = []
        for key in sorted(summary):
            s = summary[key]
            acc = s.get("accuracy")
            rows_out.append(
                {
                    "metric": key,
                    "ok": s.get("ok"),
                    "n": s.get("n"),
                    "accuracy": acc,
                    "accuracy_pct": None if acc is None else round(100 * acc, 1),
                    "random_baseline_pct": round(100 * s["random_baseline"], 1)
                    if s.get("random_baseline")
                    else None,
                }
            )
        return {
            "status": "ok",
            "path": str(path),
            "kind": kind,
            "rows": rows_out,
            "headline": _headline_multitile(rows_out),
        }

    if kind == "pairwise":
        comps = data.get("comparisons") or []
        scored = [c for c in comps if "vlm_correct" in c]
        ok = sum(1 for c in scored if c.get("vlm_correct"))
        n = len(scored)
        acc = ok / n if n else None
        return {
            "status": "ok",
            "path": str(path),
            "kind": kind,
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }

    if kind == "fewshot":
        results = data.get("results") or []
        scored = [r for r in results if "error" not in r and isinstance(r.get("result"), dict)]
        ok = sum(1 for r in scored if r.get("result", {}).get("pose_is_appropriate") is True)
        n = len(scored)
        acc = ok / n if n else None
        agree = data.get("agreement_with_human")
        headline = f"appropriate {ok}/{n}"
        if agree is not None:
            headline += f" | human-agree {agree.get('ok')}/{agree.get('n')}"
        return {
            "status": "ok",
            "path": str(path),
            "kind": kind,
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "agreement_with_human": agree,
            "headline": headline,
        }

    if kind == "google_robot":
        results = data.get("results") or []
        skipped = sum(1 for r in results if r.get("skipped"))
        scored = [r for r in results if r.get("vlm_winner") is not None]
        return {
            "status": data.get("status", "ok"),
            "path": str(path),
            "kind": kind,
            "n_total": len(results),
            "n_scored": len(scored),
            "n_skipped": skipped,
            "headline": data.get("headline", f"scored {len(scored)}, skipped {skipped}"),
            "note": data.get("note"),
        }

    return {"status": "unknown", "path": str(path)}


def _headline_multitile(rows: list[dict[str, Any]]) -> str:
    parts = []
    for r in rows:
        if r.get("accuracy_pct") is not None:
            parts.append(f"{r['metric']} {r['ok']}/{r['n']}={r['accuracy_pct']}%")
    return " | ".join(parts) if parts else "n/a"


def print_summary_table(specs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    w_id = max(4, max(len(s["id"]) for s in specs))
    print("\n" + "=" * 88)
    print("PILOT 32B SUITE — ACCURACY SUMMARY")
    print("=" * 88)
    print(f"{'#':<{w_id}}  {'experiment':<42}  {'result'}")
    print("-" * 88)
    for spec, m in zip(specs, metrics):
        status = m.get("status", "?")
        headline = m.get("headline", status)
        if status == "missing":
            headline = "MISSING JSON"
        elif status == "skipped":
            headline = m.get("note", "skipped")
        print(f"{spec['id']:<{w_id}}  {spec['title']:<42}  {headline}")
    print("=" * 88)


def human_gt_is_ok(groundtruth: str) -> bool:
    return str(groundtruth or "").strip().lower().startswith("o")


def load_consolidated_by_cue(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}
