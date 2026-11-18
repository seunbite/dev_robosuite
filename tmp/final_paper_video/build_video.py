from __future__ import annotations

import glob
import math
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
AS = ROOT / "papers/icra2027_motion_abstraction"
LLM = AS / ".codex_pptx/normalized_gifs_v5_320"
GT = AS / ".codex_pptx/balanced_15/normalized_gt_gifs_v5_320"
SYN = AS / "fig5_gallery_assets"
WAVE = ROOT / "tmp/final_paper_video/paddleclaw_big_claw_wave_cutout.gif"
RECOGNITION_PNG = AS / "figures/figure_humanoid_attr_bars.png"
OUT = ROOT / "output/video/one_grammar_many_robots_full50_174s_v5.mp4"

W, H, FPS = 1280, 720, 12
INK = "#132431"
MUTED = "#5E6D75"
TEAL = "#1F6FEB"
CORAL = "#EF6A5B"
GOLD = "#E6A23C"
DARK = "#0E2528"
PALE = "#F4F8F7"
WHITE = "#FFFFFF"
FONT_REG = "/System/Library/Fonts/Helvetica.ttc"
FONT_BOLD = "/System/Library/Fonts/Helvetica.ttc"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size, index=1 if bold else 0)


DETAIL, BODY, PAGE_TITLE, LARGE_TITLE = 12, 18, 30, 45
F16 = font(DETAIL)
F18 = font(BODY)
F22 = F26 = F32 = font(PAGE_TITLE)
F42 = F52 = font(LARGE_TITLE)
B18 = font(BODY, True)
B22 = B26 = B32 = font(PAGE_TITLE, True)
B42 = B52 = font(LARGE_TITLE, True)


def text(draw, xy, s, f, fill=INK, anchor=None, align="left", spacing=6):
    draw.multiline_text(xy, s, font=f, fill=fill, anchor=anchor, align=align, spacing=spacing)


def wrap(draw, s, f, width):
    words = s.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=f)[2] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


class GifClip:
    def __init__(self, path: Path, size: tuple[int, int], bg=WHITE):
        self.path = Path(path)
        self.frames = []
        self.durations = []
        im = Image.open(self.path)
        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            frame = im.convert("RGBA")
            fitted = ImageOps.contain(frame, size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, bg)
            canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2), fitted)
            self.frames.append(canvas)
            self.durations.append(max(0.02, im.info.get("duration", 80) / 1000))
        self.total = sum(self.durations)

    def at(self, t: float):
        if not self.frames:
            return Image.new("RGB", (1, 1), WHITE)
        t = t % self.total
        acc = 0.0
        for fr, dur in zip(self.frames, self.durations):
            acc += dur
            if t < acc:
                return fr
        return self.frames[-1]


OUT.parent.mkdir(parents=True, exist_ok=True)
proc = subprocess.Popen([
    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)
], stdin=subprocess.PIPE)


def emit(frame: Image.Image):
    proc.stdin.write(frame.convert("RGB").tobytes())


def run_scene(seconds, renderer):
    count = round(seconds * FPS)
    for i in range(count):
        emit(renderer(i / FPS, i / max(1, count - 1)))


def header(draw, title, subtitle="", dark=False):
    color = WHITE if dark else INK
    text(draw, (64, 38), title, B32, color)
    if subtitle:
        text(draw, (66, 94), subtitle, F18, "#BCD0CB" if dark else MUTED)


def paste_round(canvas, clip, t, box, radius=16):
    x, y, w, h = box
    source = clip.at(t)
    fitted = ImageOps.contain(source, (w, h), Image.Resampling.LANCZOS)
    im = Image.new("RGB", (w, h), WHITE)
    im.paste(fitted, ((w - fitted.width) // 2, (h - fitted.height) // 2))
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    canvas.paste(im, (x, y), mask)


# 1. Opening, 6 s
wave = GifClip(WAVE, (430, 520))
def intro(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im)
    text(d, (78, 170), "What makes a robot\ngesture recognizable?", B52, DARK, spacing=8)
    text(d, (82, 325), "Gesture generation across robot morphologies", F26, TEAL)
    text(d, (82, 405), "50 cues · 8 morphologies · human and LLM authors", F18, MUTED)
    paste_round(im, wave, t, (790, 75, 420, 530), 18)
    text(d, (1000, 640), "PaddleClaw says hi", B18, TEAL, anchor="mm")
    return im
run_scene(6, intro)

# 2. Motivation, 7 s
mot_a = GifClip(GT / "11_beckon_come_here__humanoid.gif", (330, 350))
mot_b = GifClip(GT / "11_beckon_come_here__roomba.gif", (330, 350))
def motivation(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Motivation", "Meaning survives changes in body shape; literal imitation often does not")
    paste_round(im, mot_a, t, (65, 165, 330, 350)); paste_round(im, mot_b, t, (425, 165, 330, 350))
    text(d, (230, 548), "Unitree G1", B18, CORAL, anchor="mm"); text(d, (590, 548), "Roomba", B18, TEAL, anchor="mm")
    text(d, (815, 190), "Same meaning\nDifferent physical evidence", B32, DARK)
    body = wrap(d, "A humanoid can use a hand. A Roomba must rebuild the cue with translation and yaw.", F22, 350)
    text(d, (815, 325), body, F22, MUTED)
    text(d, (815, 520), wrap(d, "What must remain invariant for people to recover the intended meaning?", B22, 350), B22, INK)
    return im
run_scene(7, motivation)

# 3. Three attributes, 9 s
attrs = [GifClip(GT / n, (300, 290)) for n in ["12_stop_palm_out__humanoid.gif", "08_rub_eye_tired__humanoid.gif", "01_nod_yes__humanoid.gif"]]
def attributes(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Recognition-critical attributes", "Different cues depend on different combinations")
    names = [("ORIENTATION", "Where a carrier points", TEAL), ("LOCATION", "Where the carrier sits", CORAL), ("MOVEMENT", "How the carrier changes", GOLD)]
    for i, clip in enumerate(attrs):
        x = 70 + 400 * i; paste_round(im, clip, t, (x, 160, 330, 310))
        text(d, (x + 165, 510), names[i][0], B18, names[i][2], anchor="mm")
        text(d, (x + 165, 555), names[i][1], B22, INK, anchor="mm")
    text(d, (640, 650), "Recognition depends on evidence carried by the body, not on a human-shaped skeleton.", B22, DARK, anchor="mm")
    return im
run_scene(9, attributes)

# 4. Retargeting and rematerialization, 9 s
transfer_a = GifClip(LLM / "01_nod_yes__single_arm.gif", (320, 320))
transfer_b = GifClip(LLM / "11_beckon_come_here__roomba.gif", (320, 320))
def transfer(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Retargeting and rematerialization", "Two strategies for carrying evidence across bodies")
    paste_round(im, transfer_a, t, (120, 170, 330, 330))
    paste_round(im, transfer_b, t, (830, 170, 330, 330))
    text(d, (285, 540), "RETARGETING", B18, TEAL, anchor="mm")
    text(d, (285, 590), "Map the same motion structure\nto another carrier", F18, INK, anchor="mm", align="center")
    text(d, (995, 540), "REMATERIALIZATION", B18, TEAL, anchor="mm")
    text(d, (995, 590), "Rebuild the cue from rotation,\noscillation, and approach", F18, INK, anchor="mm", align="center")
    return im
run_scene(9, transfer)

# 5. Method, 10 s
def method(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Symbolic Motion Programs", "A shared executable grammar; a morphology-specific compiler")
    items = [("1", "Morphology schema", "carriers, ranges, defaults"), ("2", "Human or LLM", "cue + YAML + five shots"), ("3", "Typed primitives", "pose / movement / path"), ("4", "Compiler", "robot-specific trajectory")]
    for i, (num, title_, body) in enumerate(items):
        x = 52 + 304 * i
        d.rounded_rectangle((x, 165, x + 260, 360), radius=18, fill="#EAF4F2" if i < 2 else "#FCEBE8")
        text(d, (x + 28, 190), num, B32, TEAL if i < 2 else CORAL)
        text(d, (x + 70, 194), title_, B22, INK)
        text(d, (x + 130, 285), body, F18, MUTED, anchor="mm", align="center")
        if i < 3: text(d, (x + 280, 260), "›", B42, "#B9C8CD", anchor="mm")
    d.rounded_rectangle((100, 430, 1180, 590), radius=18, fill=DARK)
    text(d, (140, 470), "pose(dir=front, grip=vert|palmin, z=80, x=60)\n+ move(wrist, x, [-35,+35], rep=3, spd=3.5, hold=0.1)", F18, "#E8F7F3")
    return im
run_scene(10, method)

# 6. Gallery title, 2 s
def gallery_title(t, p):
    im = Image.new("RGB", (W, H), DARK); d = ImageDraw.Draw(im)
    text(d, (640, 250), "50 cues × 5 robots", B52, WHITE, anchor="mm")
    text(d, (640, 330), "Complete LLM-authored Motion Program catalog", F18, "#C8DDD8", anchor="mm")
    text(d, (640, 420), "Unitree G1 · TIAGo · IIWA · Stretch · Roomba", B22, TEAL, anchor="mm")
    return im
run_scene(2, gallery_title)

# 7. Complete 50-cue catalog across five robots, 75 s
platforms = [("humanoid", "Unitree G1"), ("tiago", "TIAGo"), ("single_arm", "IIWA"), ("stretch", "Stretch"), ("roomba", "Roomba")]
groups = {}
for fp in sorted(LLM.glob("*.gif")):
    stem = fp.stem
    cue, platform = stem.rsplit("__", 1)
    groups.setdefault(cue, {})[platform] = fp
cue_keys = sorted(groups, key=lambda x: int(x.split("_", 1)[0]))
if len(cue_keys) != 50:
    raise RuntimeError(f"Expected 50 cue groups, found {len(cue_keys)}")
for ci, cue in enumerate(cue_keys):
    clips = [GifClip(groups[cue][p], (220, 320)) for p, _ in platforms]
    label = cue.split("_", 1)[1].replace("_", " ")
    def cue_scene(t, p, ci=ci, cue=cue, label=label, clips=clips):
        im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, 95), fill=DARK)
        text(d, (55, 48), f"{ci+1:02d}  {label}", B32, WHITE, anchor="lm")
        text(d, (1215, 48), f"{ci+1}/50", B18, "#B9D5CE", anchor="rm")
        for i, (platform, name) in enumerate(platforms):
            x = 45 + i * 246
            paste_round(im, clips[i], t, (x, 145, 220, 360), 12)
            text(d, (x + 110, 575), name, B18, TEAL if i < 4 else CORAL, anchor="mm")
        text(d, (640, 658), "Same cue, five action spaces", F18, MUTED, anchor="mm")
        return im
    run_scene(1.5, cue_scene)

# 8. Human vs LLM, 10 s
comparisons = [
    ("07_cover_yawn_sleepy__humanoid.gif", "Cover yawn", "Semantic direction can produce the wrong hand orientation."),
    ("11_beckon_come_here__roomba.gif", "Beckon on Roomba", "Human authoring adds distinctive orientation, oscillation and approach."),
]
for fname, title_, caption in comparisons:
    left = GifClip(GT / fname, (430, 430)); right = GifClip(LLM / fname, (430, 430))
    def compare(t, p, left=left, right=right, title_=title_, caption=caption):
        im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Human and LLM authoring", title_)
        paste_round(im, left, t, (110, 150, 430, 430)); paste_round(im, right, t, (740, 150, 430, 430))
        text(d, (325, 615), "HUMAN", B22, CORAL, anchor="mm"); text(d, (955, 615), "LLM", B22, TEAL, anchor="mm")
        text(d, (640, 675), caption, B18, INK, anchor="mm")
        return im
    run_scene(5, compare)

# 9. Results, 10 s — use the exact PNG included in the paper.
recognition_plot = Image.open(RECOGNITION_PNG).convert("RGB")
def results(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im)
    header(d, "Human recognition", "Human-authored Motion Programs remain more recognizable across all five robot embodiments")
    fitted = ImageOps.contain(recognition_plot, (1136, 447), Image.Resampling.LANCZOS)
    im.paste(fitted, (72 + (1136 - fitted.width) // 2, 155 + (447 - fitted.height) // 2))
    text(d, (640, 650), "Paper figure · reviewed binary recognition annotation", F16, MUTED, anchor="mm")
    return im
run_scene(10, results)

# 10. Three synthetic morphologies, 10 s (two cues)
synthetic = [("paddleclaw", "PaddleClaw"), ("spine_crab", "SpineCrab"), ("trileg_crown", "TrilegCrown")]
for cue, title_ in [("beckon_come_here.gif", "Beckon"), ("depict_flying_bird.gif", "Depict flying bird")]:
    clips = [GifClip(SYN / folder / cue, (350, 410)) for folder, _ in synthetic]
    def syn_scene(t, p, clips=clips, title_=title_):
        im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Three synthetic morphologies", f"{title_}: unfamiliar parts become communicative carriers")
        for i, ((folder, name), clip) in enumerate(zip(synthetic, clips)):
            x = 75 + 400 * i; paste_round(im, clip, t, (x, 150, 350, 430))
            text(d, (x + 175, 620), name, B18, [TEAL, CORAL, GOLD][i], anchor="mm")
        text(d, (640, 680), "The grammar stays fixed; the realization changes with morphology.", B18, INK, anchor="mm")
        return im
    run_scene(5, syn_scene)

# 11. Varied failures, 12 s
fail1 = [GifClip(LLM / "07_cover_yawn_sleepy__humanoid.gif", (360, 380)), GifClip(LLM / "32_pat_stomach__humanoid.gif", (360, 380))]
def failure_geometry(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Failure case 1", "Semantics mislead geometric direction")
    paste_round(im, fail1[0], t, (105, 160, 360, 380)); paste_round(im, fail1[1], t, (500, 160, 360, 380))
    text(d, (285, 580), "cover yawn", B18, CORAL, anchor="mm"); text(d, (680, 580), "pat stomach", B18, CORAL, anchor="mm")
    text(d, (925, 220), wrap(d, "The model identifies the target but produces the wrong hand orientation or contact geometry.", F18, 285), F18, INK)
    return im
run_scene(4, failure_geometry)

fail2 = [GifClip(LLM / "08_rub_eye_tired__single_arm.gif", (360, 380)), GifClip(LLM / "15_highfive_invite__stretch.gif", (360, 380))]
def failure_carrier(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Failure case 2", "Carrier and body-relative location")
    paste_round(im, fail2[0], t, (105, 160, 360, 380)); paste_round(im, fail2[1], t, (500, 160, 360, 380))
    text(d, (285, 580), "IIWA · rub eye", B18, TEAL, anchor="mm"); text(d, (680, 580), "Stretch · high five", B18, TEAL, anchor="mm")
    text(d, (925, 225), wrap(d, "The moving part is plausible, but the missing landmark or weak placement makes the cue ambiguous.", F18, 280), F18, INK)
    return im
run_scene(4, failure_carrier)

fail3 = [GifClip(LLM / "09_point_you__roomba.gif", (360, 380)), GifClip(LLM / "13_salute__roomba.gif", (360, 380))]
def failure_collapse(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Failure case 3", "Constrained-body collapse on Roomba")
    paste_round(im, fail3[0], t, (105, 160, 360, 380)); paste_round(im, fail3[1], t, (500, 160, 360, 380))
    text(d, (285, 580), "point you", B18, TEAL, anchor="mm"); text(d, (680, 580), "salute", B18, TEAL, anchor="mm")
    text(d, (920, 215), "34/50", B42, CORAL)
    text(d, (920, 280), "use fore-aft translation", F18, INK)
    text(d, (920, 370), "21/34", B42, GOLD)
    text(d, (920, 435), wrap(d, "add no rotation, arc, or lateral cue", F18, 290), F18, INK)
    text(d, (920, 565), wrap(d, "Distinct meanings collapse into similar trajectories.", B18, 290), B18, TEAL)
    return im
run_scene(4, failure_collapse)

# 12. Closing, 6 s
def closing(t, p):
    im = Image.new("RGB", (W, H), DARK); d = ImageDraw.Draw(im)
    text(d, (640, 170), "Recognizable robot gesture\nis a problem of evidence", B52, WHITE, anchor="mm", align="center")
    text(d, (640, 345), "Motion Programs expose orientation, location and movement\nso authors can preserve or rebuild that evidence across bodies.", F26, "#DDECE9", anchor="mm", align="center")
    d.rounded_rectangle((330, 470, 950, 477), radius=4, fill=TEAL)
    text(d, (640, 550), "Current LLMs recruit unfamiliar parts, but still miss critical geometry\nand underuse constrained bodies.", B22, "#FFD7D1", anchor="mm", align="center")
    text(d, (640, 670), "50 cues · 8 morphologies · one grammar", F18, "#AFC8C2", anchor="mm")
    return im
run_scene(6, closing)

# 13. Further directions, 8 s
def future(t, p):
    im = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(im); header(d, "Further directions", "Expressive and morphology-aware gesture generation")
    items = [
        ("01", "Affect and trait control", "Where can angry, happy, timid, or confident qualities enter without changing cue identity?"),
        ("02", "Perceptual feedback", "Use recognition models and human feedback to repair geometry before deployment."),
        ("03", "Automatic rematerialization", "Discover useful carriers and motion channels on previously unseen robot bodies."),
    ]
    for i, (num, title_, body) in enumerate(items):
        y = 165 + i * 165
        text(d, (80, y), num, B32, TEAL)
        text(d, (180, y), title_, B32, INK)
        text(d, (180, y + 55), wrap(d, body, F18, 930), F18, MUTED)
    return im
run_scene(8, future)

proc.stdin.close()
code = proc.wait()
if code:
    raise SystemExit(code)
print(OUT)
