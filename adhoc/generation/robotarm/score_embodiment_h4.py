#!/usr/bin/env python3
"""Score H4 Short-generation configs against required Human-GT core configs.

The scorer operates on parsed config trees, not raw strings. Every GT component
and attribute is required. Repetition and hold values are semantically binned:
``1`` versus ``plural`` and ``zero`` versus ``positive``.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")


def _normalized_value(key: str, value: Any) -> Any:
    leaf = key.rsplit(".", 1)[-1].lower()
    if leaf in {"repetition", "rep"}:
        try:
            return "single" if float(value) <= 1 else "plural"
        except (TypeError, ValueError):
            return str(value).lower()
    if leaf in {"hold", "hold_time"}:
        try:
            return "zero" if float(value) <= 0 else "positive"
        except (TypeError, ValueError):
            return str(value).lower()
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten one parsed component while preserving list structure."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(value[key], child))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            out[prefix] = tuple(_normalized_value(prefix, item) for item in value)
        else:
            for index, item in enumerate(value):
                out.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        out[prefix] = _normalized_value(prefix, value)
    return out


def _components(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for movement in row.get("movements") or []:
        kind = str(movement.get("type", "unknown")).lower()
        attrs = _flatten(movement.get("parameters") or {})
        result.append({"type": kind, "attrs": attrs})
    return result


def _pair_score(gt: dict[str, Any], pred: dict[str, Any]) -> float:
    if gt["type"] != pred["type"]:
        return -1.0
    ga, pa = gt["attrs"], pred["attrs"]
    keys = set(ga) | set(pa)
    overlap = sum(ga.get(key) == pa.get(key) for key in keys)
    return 1.0 + (overlap / len(keys) if keys else 1.0)


def _align(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Order-preserving alignment; same-type pairs prefer attribute overlap."""
    n, m = len(gt), len(pred)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        choice[i][0] = "up"
    for j in range(1, m + 1):
        choice[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(dp[i - 1][j], "up"), (dp[i][j - 1], "left")]
            score = _pair_score(gt[i - 1], pred[j - 1])
            if score >= 0:
                options.append((dp[i - 1][j - 1] + score, "diag"))
            dp[i][j], choice[i][j] = max(
                options, key=lambda item: (item[0], item[1] == "diag")
            )
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i or j:
        action = choice[i][j]
        if action == "diag":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "up":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    paired_gt = {i for i, _ in pairs}
    paired_pred = {j for _, j in pairs}
    return (
        pairs,
        [i for i in range(n) if i not in paired_gt],
        [j for j in range(m) if j not in paired_pred],
    )


@dataclass
class Counts:
    gt: int = 0
    pred: int = 0
    matched: int = 0


def _atom_label(component: int, key: str, value: Any) -> str:
    return f"component[{component}].{key}={value!r}"


def score_instance(gt_row: dict[str, Any], pred_row: dict[str, Any]) -> dict[str, Any]:
    gt, pred = _components(gt_row), _components(pred_row)
    pairs, missing_steps, extra_steps = _align(gt, pred)
    counts = Counts()
    missing: list[str] = []
    extra: list[str] = []

    # Component type itself is a required structural atom.
    counts.gt += len(gt)
    counts.pred += len(pred)
    counts.matched += len(pairs)
    missing.extend(f"component[{i}].type={gt[i]['type']!r}" for i in missing_steps)
    extra.extend(f"component[{j}].type={pred[j]['type']!r}" for j in extra_steps)

    for gi, pj in pairs:
        ga, pa = gt[gi]["attrs"], pred[pj]["attrs"]
        counts.gt += len(ga)
        counts.pred += len(pa)
        for key, value in ga.items():
            if key in pa and pa[key] == value:
                counts.matched += 1
            else:
                missing.append(_atom_label(gi, key, value))
        for key, value in pa.items():
            if key not in ga or ga[key] != value:
                extra.append(_atom_label(pj, key, value))

    for gi in missing_steps:
        attrs = gt[gi]["attrs"]
        counts.gt += len(attrs)
        missing.extend(_atom_label(gi, key, value) for key, value in attrs.items())
    for pj in extra_steps:
        attrs = pred[pj]["attrs"]
        counts.pred += len(attrs)
        extra.extend(_atom_label(pj, key, value) for key, value in attrs.items())

    recall = counts.matched / counts.gt if counts.gt else 1.0
    precision = counts.matched / counts.pred if counts.pred else float(not counts.gt)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    core_inclusion = not missing
    exact_match = core_inclusion and not extra
    return {
        "attribute": {
            "gt_required": counts.gt,
            "predicted": counts.pred,
            "matched": counts.matched,
            "required_recall": recall,
            "precision": precision,
            "f1": f1,
            "missing": missing,
            "extra": extra,
        },
        "instance": {
            "core_inclusion": core_inclusion,
            "exact_match": exact_match,
            "component_type_sequence_exact": [x["type"] for x in gt]
            == [x["type"] for x in pred],
        },
    }


def _index(master: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out = {}
    for entry in master:
        idx = int(entry["idx"])
        for platform, row in (entry.get("platforms") or {}).items():
            out[(idx, platform)] = row
    return out


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    samples: int = 10_000,
    seed: int = 20260723,
) -> list[float]:
    """Cue-level nonparametric 95% CI within one embodiment."""
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        if metric == "attribute_required_recall":
            matched = sum(row["attribute"]["matched"] for row in draw)
            required = sum(row["attribute"]["gt_required"] for row in draw)
            estimates.append(matched / required if required else 1.0)
        else:
            estimates.append(
                sum(bool(row["instance"][metric]) for row in draw) / len(draw)
            )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def score_masters(
    gt_master: list[dict[str, Any]], pred_master: list[dict[str, Any]]
) -> dict[str, Any]:
    gt_index, pred_index = _index(gt_master), _index(pred_master)
    rows = []
    for entry in gt_master:
        idx, cue = int(entry["idx"]), str(entry["cue"])
        for platform in PLATFORMS:
            key = (idx, platform)
            if key not in pred_index:
                raise ValueError(f"prediction missing idx={idx}, platform={platform}")
            score = score_instance(gt_index[key], pred_index[key])
            rows.append({"idx": idx, "cue": cue, "platform": platform, **score})

    aggregates: dict[str, Any] = {}
    for platform_index, platform in enumerate((*PLATFORMS, "all")):
        subset = rows if platform == "all" else [r for r in rows if r["platform"] == platform]
        gt_n = sum(r["attribute"]["gt_required"] for r in subset)
        pred_n = sum(r["attribute"]["predicted"] for r in subset)
        matched = sum(r["attribute"]["matched"] for r in subset)
        cell_recalls = [
            r["attribute"]["matched"] / r["attribute"]["gt_required"]
            if r["attribute"]["gt_required"]
            else 1.0
            for r in subset
        ]
        cell_precisions = [
            r["attribute"]["matched"] / r["attribute"]["predicted"]
            if r["attribute"]["predicted"]
            else float(not r["attribute"]["gt_required"])
            for r in subset
        ]
        cell_f1 = [
            2 * precision_i * recall_i / (precision_i + recall_i)
            if precision_i + recall_i
            else 0.0
            for precision_i, recall_i in zip(cell_precisions, cell_recalls)
        ]
        cell_jaccard = [
            r["attribute"]["matched"]
            / (
                r["attribute"]["gt_required"]
                + r["attribute"]["predicted"]
                - r["attribute"]["matched"]
            )
            if (
                r["attribute"]["gt_required"]
                + r["attribute"]["predicted"]
                - r["attribute"]["matched"]
            )
            else 1.0
            for r in subset
        ]
        recall = matched / gt_n if gt_n else 1.0
        precision = matched / pred_n if pred_n else float(not gt_n)
        aggregates[platform] = {
            "n_instances": len(subset),
            "mean_gt_attribute_count": gt_n / len(subset),
            "attribute_micro_required_recall": recall,
            "attribute_micro_required_recall_ci95": _bootstrap_ci(
                subset,
                "attribute_required_recall",
                seed=20260723 + platform_index,
            ),
            "attribute_micro_precision": precision,
            "attribute_micro_f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "attribute_macro_required_recall": sum(cell_recalls) / len(subset),
            "attribute_macro_f1": sum(cell_f1) / len(subset),
            "attribute_macro_jaccard": sum(cell_jaccard) / len(subset),
            "missing_per_100_required": 100
            * sum(len(r["attribute"]["missing"]) for r in subset)
            / gt_n,
            "extra_per_instance": sum(
                len(r["attribute"]["extra"]) for r in subset
            )
            / len(subset),
            "instance_core_inclusion_accuracy": sum(
                r["instance"]["core_inclusion"] for r in subset
            )
            / len(subset),
            "instance_core_inclusion_accuracy_ci95": _bootstrap_ci(
                subset,
                "core_inclusion",
                seed=20260733 + platform_index,
            ),
            "instance_exact_match_accuracy": sum(
                r["instance"]["exact_match"] for r in subset
            )
            / len(subset),
            "instance_exact_match_accuracy_ci95": _bootstrap_ci(
                subset,
                "exact_match",
                seed=20260743 + platform_index,
            ),
        }
    return {
        "metric_definition": {
            "unit": "cue × embodiment parsed config",
            "gt_policy": "every GT component and attribute is required core",
            "repetition": "1=single; >1=plural",
            "hold": "<=0=zero; >0=positive",
            "core_inclusion": "all GT structural and attribute atoms matched; extras allowed",
            "exact_match": "core inclusion with no extra structural or attribute atoms",
        },
        "aggregate": aggregates,
        "instances": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score_masters(
        json.loads(args.gt.read_text(encoding="utf-8")),
        json.loads(args.pred.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for platform, values in result["aggregate"].items():
        print(
            platform,
            f"attr-F1={values['attribute_micro_f1']:.3f}",
            f"core={values['instance_core_inclusion_accuracy']:.3f}",
            f"EM={values['instance_exact_match_accuracy']:.3f}",
        )


if __name__ == "__main__":
    main()
