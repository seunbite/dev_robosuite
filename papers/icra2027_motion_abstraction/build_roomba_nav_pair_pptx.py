#!/usr/bin/env python3
"""Sample PPTX from roomba_nav_eval.json: one slide per trial (A or B)."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
EVAL_JSON = PAPER / "roomba_nav_pair_sample" / "roomba_nav_eval.json"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt/roomba/gif"
LLM_INV = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/roomba/gif"
LLM_CONTRAST = REPO / "data/results/render/roomba_nav_contrastive_gemini/roomba/gif"
OUT_DIR = PAPER / "roomba_nav_pair_sample"

# GIF stays on the left; copy starts well to its right to avoid overlap.
GIF_LEFT = Inches(0.25)
GIF_WIDTH = Inches(6.0)
TEXT_LEFT = Inches(7.35)
TEXT_WIDTH = Inches(5.7)


def _question(pair: dict) -> dict:
    if "question" in pair:
        return pair["question"]
    qs = pair["questions"]
    return qs[0] if isinstance(qs, list) else qs


def _inventory_gif(root: Path, idx: int, cue: str) -> Path:
    return root / f"emb_g{idx:02d}_{cue}.gif"


def _contrast_gif(pair_id: int, ab: str, cue: str) -> Path:
    return LLM_CONTRAST / f"p{pair_id:02d}_{ab.lower()}_{cue}.gif"


def _textbox(slide, left, top, width, height, text, *, size=18, bold=False, color=RGBColor(30, 30, 30), align=PP_ALIGN.LEFT):
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


def _resolve_gif(mode: str, pair: dict, ab: str) -> Path:
    side = pair[ab.lower()]
    idx, cue = int(side["idx"]), side["cue"]
    if mode == "human":
        return _inventory_gif(GT_GIF, idx, cue)
    tagged = _contrast_gif(int(pair["id"]), ab, cue)
    if tagged.is_file():
        return tagged
    return _inventory_gif(LLM_INV, idx, cue)


def build_deck(*, mode: str, author_label: str, out_pptx: Path, spec: dict) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_blank():
        return prs.slides.add_slide(prs.slide_layouts[6])

    n_pairs = len(spec["pairs"])
    n_trials = n_pairs * 2

    s = add_blank()
    _textbox(
        s, Inches(0.6), Inches(2.1), Inches(12.1), Inches(1.0),
        f"Roomba social navigation  ·  {author_label}",
        size=28, bold=True, align=PP_ALIGN.CENTER,
    )
    _textbox(
        s, Inches(1.5), Inches(3.3), Inches(10.3), Inches(2.4),
        f"{n_pairs} pairs  ·  {n_trials} trials  ·  one slide each\n"
        "Same scene and question; only the GIF changes\n"
        "Option 1 = meaning A    Option 2 = meaning B\n"
        "Play as a slideshow so the GIFs animate.",
        size=18, align=PP_ALIGN.CENTER,
    )

    trial = 0
    for pair in spec["pairs"]:
        q = _question(pair)
        for ab, ans in (("A", 1), ("B", 2)):
            trial += 1
            s = add_blank()
            _textbox(
                s, TEXT_LEFT, Inches(0.22), TEXT_WIDTH, Inches(0.32),
                f"{trial:02d} / {n_trials}   ·   Pair {pair['id']:02d}{ab}   ·   {author_label}",
                size=12, bold=True, color=RGBColor(100, 100, 100),
            )
            _textbox(
                s, TEXT_LEFT, Inches(0.55), TEXT_WIDTH, Inches(1.55),
                pair["scene"],
                size=16, bold=True,
            )
            gif = _resolve_gif(mode, pair, ab)
            s.shapes.add_picture(str(gif), GIF_LEFT, Inches(0.85), width=GIF_WIDTH)
            _textbox(
                s, TEXT_LEFT, Inches(2.25), TEXT_WIDTH, Inches(3.5),
                f"{q['prompt']}\n\n"
                f"1)  {q['opt1']}\n\n"
                f"2)  {q['opt2']}",
                size=18,
            )
            _textbox(
                s, TEXT_LEFT, Inches(6.85), TEXT_WIDTH, Inches(0.4),
                f"key (not for participants): {ans}",
                size=11, color=RGBColor(160, 60, 60),
            )

    s = add_blank()
    _textbox(s, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.4), "Answer key (not for participants)", size=20, bold=True)
    lines = [
        f"{p['id']:02d}A  {p['a']['cue']}  →  1\n{p['id']:02d}B  {p['b']['cue']}  →  2"
        for p in spec["pairs"]
    ]
    _textbox(s, Inches(0.5), Inches(0.85), Inches(12.3), Inches(6.2), "\n\n".join(lines), size=18)

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_pptx)
    print("wrote", out_pptx, "slides", len(prs.slides))


def main() -> None:
    spec = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_deck(
        mode="human",
        author_label="Human GT",
        out_pptx=OUT_DIR / "roomba_nav_pairs_human.pptx",
        spec=spec,
    )
    build_deck(
        mode="llm",
        author_label="Gemini",
        out_pptx=OUT_DIR / "roomba_nav_pairs_llm.pptx",
        spec=spec,
    )


if __name__ == "__main__":
    main()
