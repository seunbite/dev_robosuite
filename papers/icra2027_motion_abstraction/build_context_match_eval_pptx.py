#!/usr/bin/env python3
"""Human-eval PPTX: match among restage GIFs (not a lone clip vs labels).

Same cue, same robot, different viewing contexts. Two decks:
  2AFC — two neighbor restages; pick which clip matches a named situation
  4AFC — four restages; pick which clip matches a named situation

A compact unmodified (human GT, no context) clip sits on every trial as a
reference, not as an option. Copy is English.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
GIF_ROOT = REPO / "data/results/render/social_nav_context_llm"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt"
OUT_DIR = PAPER / "context_match_eval"
SEED = 20260909

CUES = (
    (4, "gaze_shift_avoid", "gaze aversion"),
    (11, "beckon_come_here", "beckoning someone over"),
    (20, "hands_up_surrender", "hands-up surrender"),
)
PLATFORMS = ("stretch", "roomba")
PLAT_EN = {
    "stretch": "Stretch  ·  arm, head, and base",
    "roomba": "Roomba  ·  disk base only",
}

# Short English situation lines (YAML `label` is Korean).
LABEL_EN = {
    "urgent_emergency": "Shown so it can be read immediately — time pressure or nearby danger",
    "show_to_older_adult": "Shown slowly and clearly to an older adult who tracks motion slowly",
    "show_to_small_child": "Shown at a small child's eye height, without startling them",
    "show_to_seated_person": "Shown at the eye height of a seated person",
    "quiet_place": "Shown in a quiet place, without disturbing anyone",
    "crowded_passage": "Shown from above so it is not hidden among people",
    "far_away": "Shown to someone about 10 meters down the corridor",
    "carrying_load": "Shown while the robot is carrying a load, without using the arm",
    "approaching_from_behind": "Shown from behind the person — announce first so they are not startled",
}

# All viewing situations except routine (routine is never the target).
NON_ROUTINE = tuple(LABEL_EN.keys())

# One 2AFC item per target: (asked, foil). Foil may repeat; asked does not include routine.
PAIRS_2AFC = (
    ("urgent_emergency", "quiet_place"),
    ("quiet_place", "urgent_emergency"),
    ("show_to_small_child", "crowded_passage"),
    ("crowded_passage", "show_to_small_child"),
    ("show_to_seated_person", "crowded_passage"),
    ("show_to_older_adult", "urgent_emergency"),
    ("far_away", "quiet_place"),
    ("carrying_load", "crowded_passage"),
    ("approaching_from_behind", "far_away"),
)

# For each target, three contrasting foils (also non-routine).
FOILS_4AFC = {
    "urgent_emergency": ("quiet_place", "show_to_older_adult", "approaching_from_behind"),
    "show_to_older_adult": ("urgent_emergency", "quiet_place", "show_to_seated_person"),
    "show_to_small_child": ("crowded_passage", "show_to_seated_person", "far_away"),
    "show_to_seated_person": ("show_to_small_child", "crowded_passage", "show_to_older_adult"),
    "quiet_place": ("urgent_emergency", "far_away", "carrying_load"),
    "crowded_passage": ("show_to_small_child", "carrying_load", "far_away"),
    "far_away": ("quiet_place", "crowded_passage", "approaching_from_behind"),
    "carrying_load": ("crowded_passage", "quiet_place", "show_to_older_adult"),
    "approaching_from_behind": ("far_away", "urgent_emergency", "quiet_place"),
}

assert {a for a, _ in PAIRS_2AFC} == set(NON_ROUTINE)
assert len(PAIRS_2AFC) == len(NON_ROUTINE) == 9
assert set(FOILS_4AFC) == set(NON_ROUTINE)
assert "routine" not in NON_ROUTINE

WIDE, HIGH = Inches(13.333), Inches(7.5)


def _labels() -> dict[str, str]:
    return dict(LABEL_EN)


def _gif(platform: str, idx: int, cue: str, ctx: str) -> Path:
    return GIF_ROOT / platform / "gif" / f"emb_g{idx:02d}_{cue}__{ctx}.gif"


def _gt(platform: str, idx: int, cue: str) -> Path:
    return GT_GIF / platform / "gif" / f"emb_g{idx:02d}_{cue}.gif"


def _textbox(slide, left, top, width, height, text, *, size=18, bold=False, color=RGBColor(28, 28, 28), align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.alignment = align
        p.space_after = Pt(4)
    return box


def _add_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _put_gif(slide, path: Path, left, top, width, caption: str) -> None:
    if path.is_file():
        slide.shapes.add_picture(str(path), left, top, width=width)
    else:
        _textbox(slide, left, top + Inches(1.2), width, Inches(0.6), f"(missing)\n{path.name}", size=11, color=RGBColor(180, 40, 40), align=PP_ALIGN.CENTER)
    _textbox(slide, left, top - Inches(0.28), width, Inches(0.26), caption, size=12, bold=True, align=PP_ALIGN.CENTER, color=RGBColor(70, 70, 70))


def _title_slide(prs, title: str, body: str) -> None:
    s = _add_blank(prs)
    _textbox(s, Inches(0.7), Inches(1.9), Inches(12.0), Inches(1.1), title, size=28, bold=True, align=PP_ALIGN.CENTER)
    _textbox(s, Inches(1.2), Inches(3.2), Inches(10.9), Inches(3.2), body, size=16, align=PP_ALIGN.CENTER, color=RGBColor(70, 70, 70))


def _header(slide, order: int, total: int, header: str, cue_en: str, plat_en: str) -> None:
    _textbox(
        slide, Inches(0.28), Inches(0.12), Inches(10.4), Inches(0.28),
        f"{order:02d} / {total}   ·   {header}   ·   {cue_en}   ·   {plat_en}",
        size=12, bold=True, color=RGBColor(110, 110, 110),
    )


def _key_line(slide, answer: str) -> None:
    _textbox(
        slide, Inches(0.28), Inches(7.15), Inches(12.7), Inches(0.26),
        f"key (not for participants): {answer}",
        size=11, color=RGBColor(170, 50, 50),
    )


def _neutral_ref(slide, path: Path) -> None:
    """Tiny unmodified GT in the top-right. Not a response option."""
    left, top, w = Inches(11.55), Inches(0.06), Inches(1.52)
    _textbox(slide, left, top, w, Inches(0.20), "reference · unmodified", size=10, bold=True, align=PP_ALIGN.CENTER, color=RGBColor(21, 101, 192))
    if path.is_file():
        slide.shapes.add_picture(str(path), left, top + Inches(0.20), width=w)
    else:
        _textbox(slide, left, top + Inches(0.40), w, Inches(0.4), "(no GT gif)", size=10, color=RGBColor(180, 40, 40), align=PP_ALIGN.CENTER)


def _key_slide(prs, rows: list[str]) -> None:
    s = _add_blank(prs)
    _textbox(s, Inches(0.35), Inches(0.18), Inches(12.6), Inches(0.36), "Answer key  ·  not for participants", size=18, bold=True)
    if len(rows) <= 26:
        _textbox(s, Inches(0.35), Inches(0.62), Inches(12.6), Inches(6.6), "\n".join(rows), size=11, color=RGBColor(50, 50, 50))
        return
    mid = (len(rows) + 1) // 2
    _textbox(s, Inches(0.28), Inches(0.62), Inches(6.4), Inches(6.6), "\n".join(rows[:mid]), size=9, color=RGBColor(50, 50, 50))
    _textbox(s, Inches(6.75), Inches(0.62), Inches(6.4), Inches(6.6), "\n".join(rows[mid:]), size=9, color=RGBColor(50, 50, 50))


def build_2afc(labels: dict[str, str], rng: random.Random) -> tuple[Presentation, list[dict]]:
    trials = []
    for idx, cue, cue_en in CUES:
        for plat in PLATFORMS:
            for asked, other in PAIRS_2AFC:
                order = [asked, other]
                rng.shuffle(order)
                trials.append({
                    "idx": idx,
                    "cue": cue,
                    "cue_en": cue_en,
                    "platform": plat,
                    "asked": asked,
                    "foil": other,
                    "left": order[0],
                    "right": order[1],
                    "answer": 1 if order[0] == asked else 2,
                    "gif_left": str(_gif(plat, idx, cue, order[0])),
                    "gif_right": str(_gif(plat, idx, cue, order[1])),
                    "gif_neutral": str(_gt(plat, idx, cue)),
                })
    rng.shuffle(trials)
    prs = Presentation()
    prs.slide_width, prs.slide_height = WIDE, HIGH
    n = len(trials)
    _title_slide(
        prs,
        "Match the situation  ·  two clips / two choices",
        f"{n} trials  ·  same cue, same robot, two neighboring viewing situations\n"
        "Compare the left and right clips. Pick the one restaged for the situation in the question.\n"
        "Nine situations (no ordinary-corridor control as the target), each used equally as the answer.\n"
        "The small clip at top right is the unmodified reference. Do not pick it.\n"
        "Chance 50%.  Play as a slideshow so the GIFs animate.",
    )
    keys = []
    gif_w, gif_top = Inches(5.70), Inches(2.12)
    left_x, right_x = Inches(0.40), Inches(7.05)
    for i, t in enumerate(trials, 1):
        s = _add_blank(prs)
        _header(s, i, n, "2AFC · GIF matching", t["cue_en"], PLAT_EN[t["platform"]])
        _neutral_ref(s, Path(t["gif_neutral"]))
        _textbox(
            s, Inches(0.45), Inches(0.48), Inches(10.6), Inches(1.05),
            "Which clip was restaged for this situation?\n" + labels[t["asked"]],
            size=18, bold=True,
        )
        _put_gif(s, Path(t["gif_left"]), left_x, gif_top, gif_w, "A  ·  left")
        _put_gif(s, Path(t["gif_right"]), right_x, gif_top, gif_w, "B  ·  right")
        _textbox(
            s, Inches(0.45), Inches(6.55), Inches(12.4), Inches(0.50),
            "1)  left (A)          2)  right (B)",
            size=18, bold=True, align=PP_ALIGN.CENTER,
        )
        _key_line(s, f"{t['answer']}  ({'A' if t['answer']==1 else 'B'} = {t['asked']})")
        keys.append(
            f"{i:02d}  {t['platform']:7}  {t['cue']:<22}  "
            f"A={t['left']:<24}  B={t['right']:<24}  asked={t['asked']:<24}  →  {t['answer']}"
        )
    _key_slide(prs, keys)
    return prs, trials


def build_4afc(labels: dict[str, str], rng: random.Random) -> tuple[Presentation, list[dict]]:
    trials = []
    for idx, cue, cue_en in CUES:
        for plat in PLATFORMS:
            for asked in NON_ROUTINE:
                slots = [asked, *FOILS_4AFC[asked]]
                rng.shuffle(slots)
                trials.append({
                    "idx": idx,
                    "cue": cue,
                    "cue_en": cue_en,
                    "platform": plat,
                    "asked": asked,
                    "slots": slots,
                    "answer": slots.index(asked) + 1,
                    "gifs": [str(_gif(plat, idx, cue, k)) for k in slots],
                    "gif_neutral": str(_gt(plat, idx, cue)),
                })
    rng.shuffle(trials)
    prs = Presentation()
    prs.slide_width, prs.slide_height = WIDE, HIGH
    n = len(trials)
    _title_slide(
        prs,
        "Match the situation  ·  four clips / four choices",
        f"{n} trials  ·  same cue, same robot, four viewing situations\n"
        "Compare the four clips. Pick the one restaged for the situation in the question.\n"
        "Nine situations (no ordinary-corridor control as the target), each used equally as the answer.\n"
        "The small clip at top right is the unmodified reference. Do not pick it.\n"
        "Chance 25%.  Play as a slideshow so the GIFs animate.",
    )
    keys = []
    gif_w, gif_top = Inches(3.10), Inches(2.12)
    xs = [Inches(0.22 + i * 3.28) for i in range(4)]
    letters = "ABCD"
    for i, t in enumerate(trials, 1):
        s = _add_blank(prs)
        _header(s, i, n, "4AFC · GIF matching", t["cue_en"], PLAT_EN[t["platform"]])
        _neutral_ref(s, Path(t["gif_neutral"]))
        _textbox(
            s, Inches(0.28), Inches(0.48), Inches(10.8), Inches(1.15),
            "Which clip was restaged for this situation?\n" + labels[t["asked"]],
            size=18, bold=True,
        )
        for k, (ctx, gif, x) in enumerate(zip(t["slots"], t["gifs"], xs)):
            _put_gif(s, Path(gif), x, gif_top, gif_w, f"{k + 1})  {letters[k]}")
        _textbox(
            s, Inches(0.28), Inches(6.55), Inches(12.7), Inches(0.50),
            "1)  A          2)  B          3)  C          4)  D",
            size=18, bold=True, align=PP_ALIGN.CENTER,
        )
        _key_line(s, f"{t['answer']}  ({letters[t['answer']-1]} = {t['asked']})")
        slot_s = "  ".join(f"{letters[k]}={ctx}" for k, ctx in enumerate(t["slots"]))
        keys.append(
            f"{i:02d}  {t['platform']:7}  {t['cue']:<22}  asked={t['asked']:<24}  {slot_s}  →  {t['answer']}"
        )
    _key_slide(prs, keys)
    return prs, trials


def main() -> None:
    labels = _labels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    ctxs = set(NON_ROUTINE)
    for asked, foil in PAIRS_2AFC:
        ctxs.add(foil)
    for foils in FOILS_4AFC.values():
        ctxs.update(foils)
    for idx, cue, _ in CUES:
        for plat in PLATFORMS:
            gt = _gt(plat, idx, cue)
            if not gt.is_file():
                missing.append(str(gt))
            for ctx in ctxs:
                p = _gif(plat, idx, cue, ctx)
                if not p.is_file():
                    missing.append(str(p))
    if missing:
        raise SystemExit("missing GIFs:\n" + "\n".join(missing[:16]))

    prs2, t2 = build_2afc(labels, random.Random(SEED))
    p2 = OUT_DIR / "context_match_2afc.pptx"
    prs2.save(p2)
    print("wrote", p2, "slides", len(prs2.slides), "trials", len(t2))

    prs4, t4 = build_4afc(labels, random.Random(SEED + 1))
    p4 = OUT_DIR / "context_match_4afc.pptx"
    prs4.save(p4)
    print("wrote", p4, "slides", len(prs4.slides), "trials", len(t4))

    spec = {
        "seed": SEED,
        "task": "gif_matching_with_neutral_reference",
        "language": "en",
        "cues": [{"idx": i, "cue": c, "label_en": k} for i, c, k in CUES],
        "labels_en": LABEL_EN,
        "exclude_routine_as_target": True,
        "targets": list(NON_ROUTINE),
        "pairs_2afc": [list(p) for p in PAIRS_2AFC],
        "foils_4afc": {k: list(v) for k, v in FOILS_4AFC.items()},
        "trials_2afc": t2,
        "trials_4afc": t4,
    }
    spec_path = OUT_DIR / "context_match_eval_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", spec_path)
    from collections import Counter
    for name, ts in (("2afc", t2), ("4afc", t4)):
        c = Counter(t["asked"] for t in ts)
        print(name, "n", len(ts), "asked", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
