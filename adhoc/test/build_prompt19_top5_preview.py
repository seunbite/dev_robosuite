from pathlib import Path
import html
import json
import os
import re

import fire
from PIL import Image, ImageDraw, ImageSequence


REPO_ROOT = Path(__file__).resolve().parents[2]


SETS = {
    "prompt19_iconic": (
        REPO_ROOT / "data/seed/prompt19_iconic.json",
        REPO_ROOT / "data/motions/v19/IIWA",
    ),
    "prompt19_contextual": (
        REPO_ROOT / "data/seed/prompt19_contextual.json",
        REPO_ROOT / "data/motions/v19_contextual/IIWA",
    ),
    "prompt19_sophisticated_iconic": (
        REPO_ROOT / "data/seed/prompt19_sophisticated_iconic.json",
        REPO_ROOT / "data/motions/v19_sophisticated/IIWA",
    ),
    "prompt19_sophisticated_contextual": (
        REPO_ROOT / "data/seed/prompt19_sophisticated_contextual.json",
        REPO_ROOT / "data/motions/v19_sophisticated_contextual/IIWA",
    ),
}


PROMPT = (
    "You will see one PNG containing 5 robot gesture candidates, arranged left to right "
    "and labeled 1 through 5. Each candidate uses a different initial pose for the same "
    "cue. Choose the single candidate whose initial pose best matches the cue. If none "
    "of the 5 candidates is appropriate, answer NONE. Return only one token: 1, 2, 3, "
    "4, 5, or NONE."
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _latest_gif_for_cue(motion_dir: Path, cue: str) -> Path | None:
    patterns = [
        f"*_{cue}_c*_tiled.gif",
        f"*_{cue}_tiled.gif",
        f"*{cue}*_tiled.gif",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(motion_dir.glob(pattern))
    matches = sorted(set(matches), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _alpha_from_frames(frames: list[Image.Image]) -> Image.Image | None:
    rgba = [f.convert("RGBA") for f in frames]
    if not rgba:
        return None
    base = Image.new("RGBA", rgba[0].size, (255, 255, 255, 255))
    n = len(rgba)
    for i, frame in enumerate(rgba):
        alpha = int(255 * (0.15 + 0.7 * (i / max(1, n - 1))))
        mask = Image.new("L", frame.size, 0)
        mp = mask.load()
        pp = frame.convert("RGB").load()
        for y in range(frame.size[1]):
            for x in range(frame.size[0]):
                r, g, b = pp[x, y]
                if r < 245 or g < 245 or b < 245:
                    mp[x, y] = alpha
        layer = Image.new("RGBA", frame.size, (40, 40, 40, 0))
        layer.putalpha(mask)
        base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


def _build_top5_strip(gif_path: Path, out_png: Path) -> bool:
    gif = Image.open(gif_path)
    frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]
    if not frames:
        return False
    width, height = frames[0].size
    tile_w = width // 5
    alphas = []
    for i in range(5):
        tile_frames = [fr.crop((i * tile_w, 0, (i + 1) * tile_w, height)) for fr in frames]
        alpha = _alpha_from_frames(tile_frames)
        if alpha is not None:
            alphas.append(alpha)
    if len(alphas) != 5:
        return False

    label_h = 36
    strip = Image.new("RGB", (tile_w * 5, height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    for i, alpha in enumerate(alphas):
        strip.paste(alpha, (i * tile_w, 0))
        draw.text((i * tile_w + tile_w // 2 - 4, height + 8), str(i + 1), fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_png)
    return True


def _render_set(name: str, limit: int, out_dir: Path) -> Path:
    json_path, motion_dir = SETS[name]
    rows = json.loads(json_path.read_text())
    rows = sorted(rows, key=lambda r: r.get("idx", 10**9))[:limit]
    asset_dir = out_dir / f"{name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for row in rows:
        cue = row["cue"]
        idx = row["idx"]
        gif = _latest_gif_for_cue(motion_dir, cue)
        status = "ok"
        strip_rel = gif_rel = None
        if gif is None:
            status = "no_render_found"
        else:
            png = asset_dir / f"c{idx:02d}_{_slug(cue)}_top5.png"
            if _build_top5_strip(gif, png):
                strip_rel = _rel(png, out_dir)
            else:
                status = "failed_to_build_strip"
            gif_rel = _rel(gif, out_dir)
        cards.append((idx, cue, status, strip_rel, gif_rel))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(name)} top5 preview</title>",
        "</head><body style='max-width:1400px;margin:24px auto;padding:0 16px;background:#fff;'>",
        f"<h1 style='font:700 28px sans-serif'>{html.escape(name)} Top-5 Initial Pose Preview</h1>",
        "<p style='font:15px sans-serif;color:#555'>"
        "Left: top-5 alpha strip built from the tiled GIF. "
        "Below it: source tiled GIF. Right: the exact selection prompt."
        "</p>",
    ]
    for idx, cue, status, strip_rel, gif_rel in cards:
        media = ""
        if strip_rel:
            media += (
                f"<div><img src='{html.escape(strip_rel)}' "
                "style='max-width:100%;border:1px solid #ddd;border-radius:8px;'></div>"
            )
        if gif_rel:
            media += (
                f"<div style='margin-top:10px'><img src='{html.escape(gif_rel)}' "
                "style='max-width:100%;border:1px solid #ddd;border-radius:8px;'></div>"
            )
        if not media:
            media = (
                "<div style='padding:20px;border:1px solid #ddd;border-radius:8px;color:#777'>"
                "No render found</div>"
            )

        parts.append(
            "<div style='border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:16px 0;'>"
            f"<div style='font:600 18px sans-serif;margin-bottom:8px;'>c{idx} - {html.escape(cue)}</div>"
            f"<div style='font:14px sans-serif;color:#666;margin-bottom:12px;'>Status: {html.escape(status)}</div>"
            "<div style='display:grid;grid-template-columns:1.2fr 1fr;gap:16px;align-items:start;'>"
            f"<div>{media}</div>"
            "<div><div style='font:600 15px sans-serif;margin-bottom:8px;'>Prompt</div>"
            f"<pre style='white-space:pre-wrap;font:14px/1.5 Menlo,monospace;background:#f8fafc;padding:12px;border-radius:8px;border:1px solid #e5e7eb;'>{html.escape(PROMPT)}</pre>"
            "</div></div></div>"
        )
    parts.append("</body></html>")
    html_path = out_dir / f"{name}_top5_preview.html"
    html_path.write_text("".join(parts))
    return html_path


def main(limit: int = 10, open_html: bool = True):
    out_dir = REPO_ROOT / "adhoc/test/results/prompt19_top5_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SETS:
        path = _render_set(name, limit=limit, out_dir=out_dir)
        paths.append(path)
        print(path)
        if open_html:
            os.system(f"open '{path}'")
    return [str(p) for p in paths]


if __name__ == "__main__":
    fire.Fire(main)
