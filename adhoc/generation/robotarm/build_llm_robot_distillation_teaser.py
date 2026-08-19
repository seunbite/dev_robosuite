#!/usr/bin/env python3
"""Build the 2x2 LLM-to-robot abstraction teaser from one Tiago frame."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "data/results/shots/embodiment_unified/render/tiago/tiago__raising_hand_greeting__cues_yml.gif"
OUT = REPO / "data/results/drag_draw/fewshot_roundtrip/llm_robot_distillation_teaser.png"

W = 512
TOP_CROP = 28
PANEL_H = 484
HEADER = 42
GAP = 0
MARGIN = 0


def font(size: int, bold: bool = False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in paths:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def code_font(size: int, bold: bool = False):
    """VS Code-like macOS monospace face."""
    menlo = "/System/Library/Fonts/Menlo.ttc"
    if Path(menlo).is_file():
        return ImageFont.truetype(menlo, size, index=1 if bold else 0)
    return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)


YELLOW = (255, 193, 7, 255)


def sphere(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius=7):
    """Small semi-transparent blue 3-D sphere."""
    x, y = xy
    draw.ellipse((x-radius+2, y-radius+3, x+radius+3, y+radius+4), fill=(5,35,95,75))
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(47,114,254,190))
    draw.ellipse((x-radius+2, y-radius+2, x+radius-2, y+radius-2), fill=(100,160,255,105))
    draw.ellipse((x-radius*0.5, y-radius*0.52, x-radius*0.05, y-radius*0.08), fill=(255,255,255,135))


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=7, head=14):
    draw.line((*start, *end), fill=color, width=width)
    x0, y0 = start; x1, y1 = end
    import math
    a = math.atan2(y1-y0, x1-x0)
    for da in (2.55, -2.55):
        p = (x1 + head*math.cos(a+da), y1 + head*math.sin(a+da))
        draw.line((x1, y1, *p), fill=color, width=width)


def gradient_arrow(draw: ImageDraw.ImageDraw, start, end, width=9, head=13):
    """Short shaded vector with a darker base and luminous tip."""
    import math
    x0, y0 = start; x1, y1 = end
    steps = 18
    for i in range(steps):
        t0=i/steps; t1=(i+1)/steps
        a=(x0+(x1-x0)*t0, y0+(y1-y0)*t0)
        b=(x0+(x1-x0)*t1, y0+(y1-y0)*t1)
        c=(int(28+19*t1), int(75+39*t1), int(190+64*t1), 190)
        draw.line((*a,*b), fill=c, width=width)
    # narrow specular highlight reinforces the cylindrical 3-D appearance
    draw.line((x0+2,y0,x1+2,y1), fill=(255,255,255,115), width=2)
    ang=math.atan2(y1-y0,x1-x0)
    p1=(x1+head*math.cos(ang+2.55), y1+head*math.sin(ang+2.55))
    p2=(x1+head*math.cos(ang-2.55), y1+head*math.sin(ang-2.55))
    draw.polygon([(x1,y1),p1,p2], fill=(47,114,254,205))
    draw.line((x1,y1,*p1), fill=(255,255,255,110), width=2)


def label(canvas, x, y, letter, title):
    d = ImageDraw.Draw(canvas)
    text = f"{letter} {title}"
    f = font(24, False)
    box = d.textbbox((0, 0), text, font=f)
    tw = box[2] - box[0]
    d.text((x + (W-tw)/2, y+6), text, font=f, fill=(20, 20, 20))


def main():
    im = Image.open(SOURCE)
    im.seek(0)
    base = im.convert("RGB").crop((0, TOP_CROP, W, 512))
    panels = [base.copy() for _ in range(4)]

    # Coordinates below are in the cropped panel. Major controllable Tiago
    # joints are marked, including head, torso, shoulders, elbows, wrists/base.
    d = ImageDraw.Draw(panels[0], "RGBA")
    joints = [(256,230),(256,260),(256,306),(181,274),(160,286),(141,299),
              (301,274),(330,307),(344,389),(255,419)]
    for p in joints:
        sphere(d, p)

    # EE position + desired Cartesian direction.
    d = ImageDraw.Draw(panels[1], "RGBA")
    # Exact visual center between the two gripper fingers in the cropped frame.
    ee = (143, 292)
    sphere(d, ee, radius=8)
    # One-fifth the previous length, aligned with the gripper's 3-D direction.
    gradient_arrow(d, ee, (127, 255), width=9, head=13)

    # Motion Program: two white program blocks in the available upper field.
    d = ImageDraw.Draw(panels[2], "RGBA")
    blocks = [
        ((18, 22, 276, 126), ["POSE", "gripper(front, horiz,", " xyz=(60,50,38))"]),
        ((284, 22, 494, 126), ["PATH", "line(z, 0.20m,", "spd=.6, hold=1.5)"]),
    ]
    for box, lines in blocks:
        d.rounded_rectangle(box, radius=8, fill=(255,255,255,238), outline=(25,25,25,210), width=2)
        x0,y0,_,_=box
        for li, line in enumerate(lines):
            text_font = code_font(18, li == 0)
            d.text((x0+10, y0+10+li*29), line, font=text_font, fill=(25,25,25,255))
    arrow(d, (277, 73), (283, 73), (20,20,20,255), width=3, head=5)

    # EE trajectory: sampled path with a terminal arrow and start marker.
    d = ImageDraw.Draw(panels[3], "RGBA")
    pts = [(141,299),(137,278),(134,255),(133,231),(134,207),(137,183),(141,158),(145,132)]
    blue = (47, 114, 254, 238)
    d.line(pts, fill=blue, width=9, joint="curve")
    for p in pts[:-1]:
        d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4), fill=(238,244,255,245), outline=blue, width=2)
    sphere(d, pts[0], radius=8)
    arrow(d, pts[-2], pts[-1], blue, width=9, head=16)

    total_w = MARGIN*2 + W*2 + GAP
    total_h = MARGIN*2 + (HEADER+PANEL_H)*2 + GAP
    canvas = Image.new("RGB", (total_w,total_h), (255,255,255))
    titles = [
        ("(a)", "Joint-State Sequence"),
        ("(b)", "Cartesian-Pose Sequence"),
        ("(c)", "Symbolic Motion Program"),
        ("(d)", "Geometric Trajectory Sketch"),
    ]
    for i,(panel,(letter,title)) in enumerate(zip(panels,titles)):
        col=i%2; row=i//2
        x=MARGIN+col*(W+GAP); y=MARGIN+row*(HEADER+PANEL_H+GAP)
        label(canvas,x,y,letter,title)
        canvas.paste(panel,(x,y+HEADER))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
