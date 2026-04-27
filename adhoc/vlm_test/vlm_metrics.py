"""
Parse VLM JSONL lines and build accuracy / keyword summaries for run_exp.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _norm(s: str) -> str:
    return (s or "").lower()


def parse_yes_no(response: str) -> str | None:
    t = (response or "").strip()
    if t.startswith("ERROR:"):
        return None
    low = t.lower()
    for w in ("yes", "no"):
        if re.search(rf"\b{w}\b", low):
            return w
    if "yes" in low and "no" not in low:
        return "yes"
    if "no" in low and "yes" not in low:
        return "no"
    m = re.search(r"\b(y|n)\b", low)
    if m and m.group(1) == "y":
        return "yes"
    if m and m.group(1) == "n":
        return "no"
    return None


def parse_letter_abcd(response: str) -> str | None:
    t = (response or "").strip()
    if t.startswith("ERROR:"):
        return None
    m = re.search(r"\b([ABCD])\b", t.upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:option|answer|letter)\s*[:=]?\s*([ABCD])", t, re.I)
    if m:
        return m.group(1).upper()
    return None


def parse_letters_multiselect(response: str) -> set[str]:
    """Distinct A–D letters mentioned anywhere in the response."""
    t = (response or "").strip()
    if t.startswith("ERROR:"):
        return set()
    return set(re.findall(r"\b([ABCD])\b", t.upper()))


def compare_keyword_counts(response: str) -> dict[str, bool]:
    """Keyword presence in raw text (for compare_baseline post-hoc)."""
    low = _norm(response)
    if low.startswith("error:") or not low:
        return {"sophisticated": False, "no_reasoning": False, "neither": True}
    soph = bool(
        re.search(r"\bsophisticated\b", low)
        or re.search(r"\bsoph\b", low)
    )
    nr = bool(
        re.search(r"no[-\s]?reasoning", low)
        or re.search(r"no reason", low)
        or re.search(r"\bnr\b", low)
    )
    neither = not soph and not nr
    return {"sophisticated": soph, "no_reasoning": nr, "neither": neither}


def _binary_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def build_binary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows keyed by (robot, input_type)."""
    by: dict[tuple[str, str], list[tuple[str | None, str]]] = defaultdict(list)
    for r in records:
        if r.get("task") != "binary_classification":
            continue
        gt = (r.get("ground_truth") or {}).get("answer", "").lower()
        if gt not in ("yes", "no"):
            continue
        pred = parse_yes_no(r.get("response", ""))
        by[(r.get("robot", ""), r.get("input_type", ""))].append((pred, gt))

    rows: list[dict[str, Any]] = []
    for (robot, ity), pairs in sorted(by.items()):
        n = len(pairs)
        correct = 0
        tp = fp = fn = tn = 0
        for pred, gt in pairs:
            if pred is not None and pred == gt:
                correct += 1
            if pred is None:
                continue
            if gt == "yes" and pred == "yes":
                tp += 1
            if gt == "no" and pred == "yes":
                fp += 1
            if gt == "yes" and pred == "no":
                fn += 1
            if gt == "no" and pred == "no":
                tn += 1
        acc = correct / n if n else 0.0
        prec, rec, f1 = _binary_prf(tp, fp, fn)
        rows.append(
            {
                "robot": robot,
                "input_type": ity,
                "n": n,
                "accuracy": acc,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision_yes": prec,
                "recall_yes": rec,
                "f1_yes": f1,
            }
        )
    return rows


def build_compare_multiselect_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Multiselect: require sophisticated letter ∈ predicted set; report recall + avg k."""
    by: dict[str, list[tuple[set[str], str]]] = defaultdict(list)
    for r in records:
        if r.get("task") != "compare_baseline":
            continue
        gt_letter = (r.get("ground_truth") or {}).get("soph_letter", "").upper()
        if not gt_letter or gt_letter not in "ABCD":
            continue
        pred = parse_letters_multiselect(r.get("response", ""))
        by[r.get("robot", "")].append((pred, gt_letter))
    rows: list[dict[str, Any]] = []
    for robot, pairs in sorted(by.items()):
        n = len(pairs)
        soph_included = sum(1 for s, g in pairs if g in s)
        avg_k = sum(len(s) for s, _ in pairs) / n if n else 0.0
        rows.append(
            {
                "robot": robot,
                "n": n,
                "soph_letter_included_recall": soph_included / n if n else 0.0,
                "soph_included_count": soph_included,
                "avg_letters_picked": avg_k,
            }
        )
    return rows


def build_compare_keyword_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Global counts on compare_baseline text (not per robot, but can filter)."""
    sub = [r for r in records if r.get("task") == "compare_baseline"]
    n = len(sub)
    n_soph = n_nr = n_nei = 0
    n_both = 0
    for r in sub:
        k = compare_keyword_counts(r.get("response", ""))
        if k["sophisticated"]:
            n_soph += 1
        if k["no_reasoning"]:
            n_nr += 1
        if k["neither"]:
            n_nei += 1
        if k["sophisticated"] and k["no_reasoning"]:
            n_both += 1
    return {
        "n": n,
        "mentions_sophisticated": n_soph,
        "mentions_no_reasoning": n_nr,
        "mentions_neither_keyword": n_nei,
        "mentions_both_keywords": n_both,
    }


def format_markdown_report(records: list[dict[str, Any]], task: str) -> str:
    if not records:
        return "_No API records (dry run or empty)._"
    lines: list[str] = ["# VLM run summary", ""]
    if task == "binary_classification":
        lines.append("## binary_classification (by robot × input_type)")
        lines.append("")
        lines.append(
            "| robot | input_type | n | acc | P(yes) | R(yes) | F1(yes) | TP | FP | FN | TN |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for x in build_binary_rows(records):
            lines.append(
                f"| {x['robot']} | {x['input_type']} | {x['n']} | {x['accuracy']:.3f} | "
                f"{x['precision_yes']:.3f} | {x['recall_yes']:.3f} | {x['f1_yes']:.3f} | "
                f"{x['tp']} | {x['fp']} | {x['fn']} | {x['tn']} |"
            )
    elif task == "compare_baseline":
        lines.append("## compare_baseline: multiselect (ground truth: must include **soph** letter)")
        lines.append("")
        lines.append("| robot | n | soph included recall | n with soph in set | avg |P| letters |")
        lines.append("|---|---:|---:|---:|---:|")
        for x in build_compare_multiselect_rows(records):
            lines.append(
                f"| {x['robot']} | {x['n']} | {x['soph_letter_included_recall']:.3f} | "
                f"{x['soph_included_count']} | {x['avg_letters_picked']:.2f} |"
            )
        k = build_compare_keyword_summary(records)
        lines.append("")
        lines.append("## compare_baseline: response text keyword hits (all instances)")
        lines.append("")
        lines.append("| metric | count |")
        lines.append("|---|---:|")
        lines.append(f"| n | {k['n']} |")
        lines.append(
            f"| responses mentioning **sophisticated** / *soph* | {k['mentions_sophisticated']} |"
        )
        lines.append(
            f"| responses mentioning **no-reasoning** (incl. nr / no reason) | {k['mentions_no_reasoning']} |"
        )
        lines.append(
            f"| responses with **neither** keyword (no soph + no nr terms above) | {k['mentions_neither_keyword']} |"
        )
        lines.append(
            f"| responses mentioning **both** soph. and nr. terms | {k['mentions_both_keywords']} |"
        )
    lines.append("")
    return "\n".join(lines)
