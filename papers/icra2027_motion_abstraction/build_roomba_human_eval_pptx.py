#!/usr/bin/env python3
"""Build a Roomba social-navigation practice PPTX (GIF + A/B action choice)."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt/roomba/gif"
OUT_DIR = PAPER / "roomba_human_eval_sample"
OUT_PPTX = PAPER / "roomba_human_eval_sample.pptx"
SEED = 20260831

# Practice trials: ambiguous social-nav scene + Roomba GT signal + A/B actions.
# Primary ask = where YOU should go; secondary = where the ROBOT seems headed.
# Answers are the intended navigation reading of the cue (not cue naming).
TRIALS_SPEC = [
    {
        "idx": 12,
        "cue": "stop_palm_out",
        "contrast": "Stop vs continue",
        "scene": (
            "Narrow doorway. You and the Roomba both need to pass. "
            "The robot pauses in the doorway and signals, then the clip ends "
            "(before either of you actually crosses)."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Stop and wait outside the doorway", "Keep going through the doorway"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Hold position / block the doorway", "Drive through the doorway now"),
        "robot_answer": 0,
    },
    {
        "idx": 11,
        "cue": "beckon_come_here",
        "contrast": "Approach vs wait",
        "scene": (
            "Shared corridor. The Roomba is ahead of you near a meeting point. "
            "It signals, then the clip ends before either of you commits to a path."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Approach the robot", "Stay where you are and wait"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Stay / wait for you to come closer", "Drive away down the corridor"),
        "robot_answer": 0,
    },
    {
        "idx": 14,
        "cue": "raise_index_one_moment",
        "contrast": "Wait a moment vs go ahead",
        "scene": (
            "T-intersection. You were about to enter the shared lane. "
            "The Roomba signals from the side, then the clip ends."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Pause here for a moment", "Enter the intersection now"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Hold briefly, then proceed", "Rush through the intersection immediately"),
        "robot_answer": 0,
    },
    {
        "idx": 19,
        "cue": "arms_open_welcome",
        "contrast": "Enter / approach vs stay back",
        "scene": (
            "Doorway into a room. The Roomba is just inside facing you and signals. "
            "Clip ends before anyone crosses the threshold."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Enter / approach the robot", "Stay outside and keep distance"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Stay welcoming near the doorway", "Retreat deeper into the room to block you"),
        "robot_answer": 0,
    },
    {
        "idx": 4,
        "cue": "gaze_shift_avoid",
        "contrast": "Yield space vs claim the lane",
        "scene": (
            "Narrow hallway. You and the Roomba face each other with room to pass on one side. "
            "The robot signals with a gaze/heading shift, then the clip ends."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": (
            "Take the opening and pass (robot is yielding attention/space)",
            "Hold center and force the robot to reverse",
        ),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Turn attention / path away (yield)", "Drive straight into you"),
        "robot_answer": 0,
    },
    {
        "idx": 9,
        "cue": "point_you",
        "contrast": "You go vs I go",
        "scene": (
            "Fork in a corridor: left path and right path both work. "
            "The Roomba faces you and points/signals, then the clip ends "
            "before it rolls down either path."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": (
            "You are being addressed / cleared to move (take your turn)",
            "Freeze — the robot is claiming both paths",
        ),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Hold and direct you", "Immediately claim the path for itself"),
        "robot_answer": 0,
    },
    {
        "idx": 15,
        "cue": "highfive_invite",
        "contrast": "Approach for contact vs keep distance",
        "scene": (
            "Open lobby. The Roomba is a few meters ahead and signals an invitation. "
            "Clip ends before either of you closes the gap."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Approach the robot for contact/interaction", "Keep distance and walk past"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Come toward / meet you", "Back away to avoid you"),
        "robot_answer": 0,
    },
    {
        "idx": 20,
        "cue": "hands_up_surrender",
        "contrast": "You first vs robot first",
        "scene": (
            "Bottleneck passage. Only one can go at a time. "
            "The Roomba signals with a surrender/hands-up style motion, then freezes."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("You go first through the passage", "Wait and let the robot go first"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Yield / stay put so you can pass", "Push through first"),
        "robot_answer": 0,
    },
    {
        "idx": 2,
        "cue": "shake_head_no",
        "contrast": "Reject a proposed path vs accept",
        "scene": (
            "You were about to take the left doorway. The Roomba is near the fork and signals. "
            "Clip ends before it rolls left or right."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Do NOT take that proposed left path — pick the other", "Commit to the left doorway"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Not endorsing your left choice (may take / clear the other)", "Leading you left"),
        "robot_answer": 0,
    },
    {
        "idx": 1,
        "cue": "nod_yes",
        "contrast": "Confirm a proposed path vs reject",
        "scene": (
            "Same fork. You glance toward the right doorway as a candidate. "
            "The Roomba signals, then the clip ends."
        ),
        "you_prompt": "Where should YOU go next?",
        "you_choices": ("Take the proposed right path", "Abandon it and reverse"),
        "you_answer": 0,
        "robot_prompt": "Where does the ROBOT seem headed / intending to go?",
        "robot_choices": ("Affirming / clearing that path with you", "Blocking that path"),
        "robot_answer": 0,
    },
]


def _shuffle_ab(rng: random.Random, choices: tuple[str, str], answer: int):
    items = list(choices)
    if rng.random() < 0.5:
        items.reverse()
        answer = 1 - answer
    return items, answer


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gifs_dir = OUT_DIR / "gifs"
    gifs_dir.mkdir(exist_ok=True)
    # drop old strip PNGs if present
    strips = OUT_DIR / "strips"
    if strips.is_dir():
        shutil.rmtree(strips)

    rng = random.Random(SEED)
    trials = []
    for spec in TRIALS_SPEC:
        gif = GT_GIF / f"emb_g{spec['idx']:02d}_{spec['cue']}.gif"
        if not gif.is_file():
            print("missing", gif)
            continue
        local_gif = gifs_dir / gif.name
        shutil.copy2(gif, local_gif)

        you_choices, you_ans = _shuffle_ab(rng, spec["you_choices"], spec["you_answer"])
        robot_choices, robot_ans = _shuffle_ab(
            rng, spec["robot_choices"], spec["robot_answer"]
        )
        trials.append(
            {
                "idx": spec["idx"],
                "cue": spec["cue"],
                "contrast": spec["contrast"],
                "scene": spec["scene"],
                "gif": str(local_gif),
                "gif_name": gif.name,
                "you_prompt": spec["you_prompt"],
                "you_choices": you_choices,
                "you_answer_index": you_ans,
                "robot_prompt": spec["robot_prompt"],
                "robot_choices": robot_choices,
                "robot_answer_index": robot_ans,
            }
        )

    (OUT_DIR / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_title(slide, text, top=0.2, size=26):
        box = slide.shapes.add_textbox(Inches(0.45), Inches(top), Inches(12.4), Inches(0.5))
        tf = box.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(size)
        p.font.bold = True
        p.font.color.rgb = RGBColor(30, 30, 30)

    def add_body(slide, text, top, height=1.2, size=16, left=0.45, width=12.4):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        tf.clear()
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size = Pt(size)
            p.font.color.rgb = RGBColor(40, 40, 40)
            p.space_after = Pt(4)

    # --- Title ---
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Roomba Social Navigation — Practice Sample", top=2.0, size=30)
    add_body(
        slide,
        "Not cue naming.\n"
        "Watch the Roomba signal in an ambiguous hallway / doorway / fork scene,\n"
        "then answer: Where should I go?  /  Where will the robot go?\n"
        f"{len(trials)} trials · GT Roomba Motion Code · A/B action choice (chance 50%)\n"
        "Play as slideshow so the embedded GIFs animate. Answer key at the end.",
        top=2.9,
        height=3.0,
        size=18,
    )

    # --- Instructions ---
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Instructions")
    add_body(
        slide,
        "1. Read the scene. Two actions are feasible; the gesture should tip the choice.\n"
        "2. Watch the Roomba GIF (slideshow mode required for animation).\n"
        "3. Answer BOTH questions with A or B:\n"
        "      • Where should YOU go next?\n"
        "      • Where does the ROBOT seem headed?\n"
        "4. The clip ends before locomotion reveals the answer — decide from the signal.\n"
        "5. Write answers on paper, then check the key (cue labels are only for scoring).",
        top=1.1,
        height=5.0,
        size=20,
    )

    letters = "AB"
    for t_i, trial in enumerate(trials, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title(
            slide,
            f"Trial {t_i}/{len(trials)}  ·  Roomba  ·  {trial['contrast']}",
            top=0.12,
            size=22,
        )
        add_body(slide, f"Scene: {trial['scene']}", top=0.58, height=1.05, size=14)

        # Embedded animated GIF (plays in PowerPoint / Keynote slideshow)
        slide.shapes.add_picture(
            trial["gif"],
            Inches(0.55),
            Inches(1.7),
            height=Inches(3.55),
        )

        q_left = 5.4
        add_body(
            slide,
            trial["you_prompt"]
            + "\n"
            + f"A.  {trial['you_choices'][0]}\n"
            + f"B.  {trial['you_choices'][1]}",
            top=1.75,
            height=2.0,
            size=15,
            left=q_left,
            width=7.5,
        )
        add_body(
            slide,
            trial["robot_prompt"]
            + "\n"
            + f"A.  {trial['robot_choices'][0]}\n"
            + f"B.  {trial['robot_choices'][1]}",
            top=4.0,
            height=2.0,
            size=15,
            left=q_left,
            width=7.5,
        )
        add_body(
            slide,
            "Write: You = __   Robot = __",
            top=6.7,
            height=0.4,
            size=14,
            left=q_left,
            width=7.5,
        )

    # --- Answer key ---
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Answer key (navigation reading — not for participants)")
    lines = []
    for t_i, trial in enumerate(trials, start=1):
        ya = letters[trial["you_answer_index"]]
        ra = letters[trial["robot_answer_index"]]
        lines.append(
            f"Trial {t_i}: You={ya}  Robot={ra}  "
            f"[{trial['cue'].replace('_', ' ')}]  ·  {trial['contrast']}"
        )
        lines.append(
            f"    You: {trial['you_choices'][trial['you_answer_index']]}"
        )
        lines.append(
            f"    Robot: {trial['robot_choices'][trial['robot_answer_index']]}"
        )
    add_body(slide, "\n".join(lines), top=0.75, height=6.4, size=12)

    prs.save(OUT_PPTX)
    print("wrote", OUT_PPTX)
    print("assets", OUT_DIR)
    print("n trials", len(trials))


if __name__ == "__main__":
    build()
