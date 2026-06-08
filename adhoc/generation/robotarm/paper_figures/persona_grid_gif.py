"""Combine 10 persona GIFs per cue into a single labeled subplot GIF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise SystemExit("Pillow required: pip install pillow") from e

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Each column: high/contrast persona on top, paired low/opposite below.
PERSONA_PAIRS = (
    ("high_extraversion", "low_extraversion"),
    ("high_neuroticism", "low_neuroticism"),
    ("playful_child", "calm_steady"),
    ("aggressive", "sad_subdued"),
    ("grandmotherly", "buzz_lightyear"),
)

FONT_SIZE = 26


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold
        else ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc")
    )
    for name in names:
        for root in ("/System/Library/Fonts/Supplemental", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = Path(root) / name
            if p.is_file():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass
    return ImageFont.load_default()


def _persona_label(key: str) -> str:
    return key.replace("_", " ")


def _load_frames(path: Path) -> tuple[list[Image.Image], int]:
    im = Image.open(path)
    duration = int(im.info.get("duration", 100) or 100)
    n = getattr(im, "n_frames", 1)
    frames: list[Image.Image] = []
    for i in range(n):
        im.seek(i)
        frames.append(im.convert("RGB"))
    return frames, duration


def _paired_persona_order(personas: list[str]) -> list[str]:
    """Row-major 2×5 grid: all tops left→right, then paired bottoms in same columns."""
    ps = set(personas)
    tops: list[str] = []
    bots: list[str] = []
    for top, bot in PERSONA_PAIRS:
        if top in ps:
            tops.append(top)
        if bot in ps:
            bots.append(bot)
    extra = sorted(ps - set(tops) - set(bots))
    return tops + bots + extra


def _cue_groups(configs: list[dict[str, Any]]) -> list[tuple[int, str, list[str]]]:
    by_cue: dict[tuple[int, str], list[str]] = {}
    for cfg in configs:
        idx = int(cfg["idx"])
        cue = str(cfg["cue"])
        persona = str(cfg.get("persona", ""))
        by_cue.setdefault((idx, cue), []).append(persona)
    out: list[tuple[int, str, list[str]]] = []
    for (idx, cue), personas in sorted(by_cue.items(), key=lambda x: x[0][0]):
        out.append((idx, cue, _paired_persona_order(personas)))
    return out


def build_cue_grid_gif(
    *,
    idx: int,
    cue: str,
    personas: list[str],
    gif_dir: Path,
    out_path: Path,
    ncol: int = 5,
    panel_size: int = 220,
    persona_bar: int = 52,
    cue_bar: int = 52,
    gap: int = 6,
    margin: int = 10,
) -> Path:
    nrow = (len(personas) + ncol - 1) // ncol
    cell_w = panel_size
    cell_h = panel_size + persona_bar
    grid_w = ncol * cell_w + (ncol - 1) * gap
    grid_h = nrow * cell_h + (nrow - 1) * gap
    canvas_w = grid_w + 2 * margin
    canvas_h = grid_h + cue_bar + margin * 2

    sources: list[tuple[str, list[Image.Image], int]] = []
    for persona in personas:
        gif_path = gif_dir / f"c{idx}_{cue}_{persona}.gif"
        if not gif_path.is_file():
            raise FileNotFoundError(gif_path)
        frames, dur = _load_frames(gif_path)
        thumb = [f.resize((panel_size, panel_size), Image.Resampling.LANCZOS) for f in frames]
        sources.append((persona, thumb, dur))

    n_frames = max(len(f) for _, f, _ in sources)
    duration = int(sum(d for _, _, d in sources) / len(sources))
    duration = max(50, min(duration, 120))

    font = _font(FONT_SIZE, bold=True)

    out_frames: list[Image.Image] = []
    for t in range(n_frames):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 252))
        draw = ImageDraw.Draw(canvas)

        for k, (persona, frames, _) in enumerate(sources):
            row, col = divmod(k, ncol)
            x0 = margin + col * (cell_w + gap)
            y0 = margin + row * (cell_h + gap)
            frame = frames[min(t, len(frames) - 1)]
            canvas.paste(frame, (x0, y0))
            label = _persona_label(persona)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (panel_size - tw) // 2
            ty = y0 + panel_size + 6
            draw.text((tx, ty), label, fill=(30, 30, 40), font=font)

        cue_text = f"c{idx}  {cue.replace('_', ' ')}"
        bbox = draw.textbbox((0, 0), cue_text, font=font)
        tw = bbox[2] - bbox[0]
        cx = (canvas_w - tw) // 2
        cy = margin + grid_h + (cue_bar - (bbox[3] - bbox[1])) // 2
        draw.text((cx, cy), cue_text, fill=(10, 10, 20), font=font)

        out_frames.append(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return out_path


def run(
    *,
    persona_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    ncol: int = 5,
    panel_size: int = 220,
) -> list[Path]:
    base = Path(persona_dir) if persona_dir else _REPO / "data/results/paper_figures/persona_2cue10"
    gif_dir = base / "gifs"
    cfg_path = base / "persona_configs.json"
    od = Path(out_dir) if out_dir else base / "grid_gifs"
    od.mkdir(parents=True, exist_ok=True)

    configs = json.loads(cfg_path.read_text(encoding="utf-8"))
    groups = _cue_groups(configs)
    written: list[Path] = []

    for idx, cue, personas in groups:
        out_path = od / f"c{idx}_{cue}_persona_grid.gif"
        print(f"[grid] c{idx} {cue} ({len(personas)} personas) → {out_path}", flush=True)
        build_cue_grid_gif(
            idx=idx,
            cue=cue,
            personas=personas,
            gif_dir=gif_dir,
            out_path=out_path,
            ncol=ncol,
            panel_size=panel_size,
        )
        written.append(out_path)

    print(f"[done] {len(written)} grid GIFs → {od}", flush=True)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Persona 10-panel subplot GIF per cue")
    p.add_argument("--persona-dir", default=None, help="persona_2cue10 output dir")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ncol", type=int, default=5)
    p.add_argument("--panel-size", type=int, default=220)
    args = p.parse_args()
    run(persona_dir=args.persona_dir, out_dir=args.out_dir, ncol=args.ncol, panel_size=args.panel_size)


if __name__ == "__main__":
    main()
