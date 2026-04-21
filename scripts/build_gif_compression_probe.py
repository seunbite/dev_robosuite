from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageSequence


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
OUT_ROOT = ROOT / "data" / "seed" / "gif_compression_probe"

SAMPLES = [
    ("iconic", "raising_hand_greeting", ROOT / "data" / "motions" / "v19_sophisticated" / "IIWA" / "20260409_064025_IIWA_raising_hand_greeting_p615.gif"),
    ("iconic", "fan_face_hot", ROOT / "data" / "motions" / "v19_sophisticated" / "IIWA" / "20260409_064101_IIWA_fan_face_hot_p275.gif"),
    ("iconic", "firm_accept_forward_reach", ROOT / "data" / "motions" / "v19_sophisticated" / "IIWA" / "20260409_064007_IIWA_firm_accept_forward_reach_p275.gif"),
    ("contextual", "nod_yes", ROOT / "data" / "motions" / "v19_sophisticated_contextual" / "IIWA" / "20260409_064202_IIWA_nod_yes_p35.gif"),
    ("contextual", "self_hug", ROOT / "data" / "motions" / "v19_sophisticated_contextual" / "IIWA" / "20260409_064207_IIWA_self_hug_p35.gif"),
    ("contextual", "listening_lean_substitute", ROOT / "data" / "motions" / "v19_sophisticated_contextual" / "IIWA" / "20260409_064330_IIWA_listening_lean_substitute_p35.gif"),
]

VARIANTS = [
    ("v1_420w_128c", 420, 128),
    ("v2_360w_96c", 360, 96),
    ("v3_320w_64c", 320, 64),
]


def iter_frames(path: Path) -> tuple[list[Image.Image], list[int], int]:
    img = Image.open(path)
    frames = []
    durations = []
    loop = img.info.get("loop", 0)
    for frame in ImageSequence.Iterator(img):
        frames.append(frame.convert("RGBA"))
        durations.append(frame.info.get("duration", img.info.get("duration", 120)))
    return frames, durations, loop


def quantize_frame(frame: Image.Image, max_width: int, colors: int) -> Image.Image:
    scale = min(1.0, max_width / frame.width)
    new_size = (max(1, int(frame.width * scale)), max(1, int(frame.height * scale)))
    resized = frame.resize(new_size, Image.Resampling.LANCZOS)
    white = Image.new("RGBA", resized.size, (255, 255, 255, 255))
    white.alpha_composite(resized)
    rgb = white.convert("RGB")
    return rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def write_variant(src: Path, dst: Path, max_width: int, colors: int) -> dict:
    frames, durations, loop = iter_frames(src)
    qframes = [quantize_frame(frame, max_width, colors) for frame in frames]
    dst.parent.mkdir(parents=True, exist_ok=True)
    qframes[0].save(
        dst,
        save_all=True,
        append_images=qframes[1:],
        duration=durations,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    with Image.open(dst) as img:
        width, height = img.size
        nframes = getattr(img, "n_frames", 1)
        delay = img.info.get("duration", 0)
    return {
        "path": dst,
        "size_bytes": dst.stat().st_size,
        "width": width,
        "height": height,
        "frames": nframes,
        "delay_ms": delay,
        "colors": colors,
    }


def file_meta(path: Path) -> dict:
    with Image.open(path) as img:
        width, height = img.size
        nframes = getattr(img, "n_frames", 1)
        delay = img.info.get("duration", 0)
    return {
        "path": path,
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "frames": nframes,
        "delay_ms": delay,
    }


def human_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def rel(path: Path) -> str:
    return path.relative_to(OUT_ROOT).as_posix()


def render_html(rows: Iterable[dict]) -> str:
    cards = []
    for row in rows:
        blocks = []
        all_variants = [row["original"], *row["variants"]]
        for variant in all_variants:
            title = variant["label"]
            meta = f'{variant["width"]}x{variant["height"]}, {variant["frames"]}f, {variant["delay_ms"]}ms, {human_mb(variant["size_bytes"])}'
            if "colors" in variant:
                meta += f', {variant["colors"]} colors'
            blocks.append(
                f"""
                <div class="variant">
                  <div class="vtitle">{title}</div>
                  <img src="{rel(variant['path'])}" />
                  <div class="meta">{meta}</div>
                </div>
                """
            )
        cards.append(
            f"""
            <section class="card">
              <div class="chead">
                <div class="eyebrow">{row['dataset']}</div>
                <h2>{row['cue']}</h2>
              </div>
              <div class="variants">{''.join(blocks)}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GIF Compression Probe</title>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #111111;
      --muted: #5f6670;
      --line: #d9dfe6;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{
      width: min(1600px, calc(100vw - 48px));
      margin: 24px auto 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .sub {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 15px;
    }}
    .card {{
      border-top: 1px solid var(--line);
      padding: 18px 0 22px;
    }}
    .chead {{
      margin-bottom: 14px;
    }}
    .chead h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .variants {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .variant {{
      border: 1px solid var(--line);
      padding: 10px;
      background: #fff;
    }}
    .vtitle {{
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .variant img {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
    }}
    .meta {{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>GIF Compression Probe</h1>
    <p class="sub">FPS is unchanged. Only resolution and palette size are reduced. Compare quality and file size before choosing a PPT pipeline.</p>
    {''.join(cards)}
  </div>
</body>
</html>"""


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, cue, src in SAMPLES:
        cue_dir = OUT_ROOT / f"{dataset}_{cue}"
        cue_dir.mkdir(parents=True, exist_ok=True)
        original_dst = cue_dir / "original.gif"
        if not original_dst.exists():
            original_dst.write_bytes(src.read_bytes())
        row = {
            "dataset": dataset,
            "cue": cue,
            "original": {
                "label": "Original",
                **file_meta(original_dst),
            },
            "variants": [],
        }
        for label, max_width, colors in VARIANTS:
            dst = cue_dir / f"{label}.gif"
            meta = write_variant(src, dst, max_width=max_width, colors=colors)
            meta["label"] = label.replace("_", " ")
            row["variants"].append(meta)
        rows.append(row)

    html_path = OUT_ROOT / "gif_compression_probe_20260412.html"
    html_path.write_text(render_html(rows), encoding="utf-8")
    print("WROTE", html_path)
    for row in rows:
        print(row["dataset"], row["cue"], human_mb(row["original"]["size_bytes"]), "->", [human_mb(v["size_bytes"]) for v in row["variants"]])


if __name__ == "__main__":
    main()
