#!/usr/bin/env python3
"""Per-tile Gemini plausibility check for one Tiago pose group (e.g. front+horizontal)."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _HERE, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_pose_group_tiles import (  # noqa: E402
    DEFAULT_GRID_COLS,
    DEFAULT_HEAD_HAND_GAP_MIN_M,
    ORDER_DIR,
    ORDER_GRIP,
    SOURCE_DIR_FOR_DISPLAY,
    _select_xy_grid_balanced,
)
from legacy.render_mobile_config import (  # noqa: E402
    QI_R_ARM,
    _capture,
    _default_qpos,
    _get_pose_bank_entries,
    _make_env,
    _right_hand_forward_vs_torso_head_ok,
)

PROMPT_PATH = _REPO / "data/seed/prompt/tiago/prompt_verify_pose_plausibility.txt"
DEFAULT_GROUP = "front_horizontal"
DEFAULT_OUT = _REPO / "data/results/verify/tiago/pose_plausibility_front_horizontal.json"
DEFAULT_VIS = _REPO / "data/results/visualize/tiago/pose_plausibility_front_horizontal"

MODEL_FALLBACK = ("gemini-2.5-pro", "gemini-2.5-flash")


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _gemini_client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (or source APIKEY.sh).")
    return genai.Client(api_key=key)


def _generate_with_fallback(client, *, model: str, img_bytes: bytes, prompt: str) -> tuple[str, str]:
    from google.genai import types

    models = [model] if model else list(MODEL_FALLBACK)
    if model and model not in MODEL_FALLBACK:
        models = [model, *MODEL_FALLBACK]
    else:
        seen: set[str] = set()
        models = [m for m in models if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

    last_err: Exception | None = None
    for m in models:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            return (resp.text or "").strip(), m
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                print(f"[warn] {m} exhausted, trying fallback...", flush=True)
                continue
            raise
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def _parse_group_key(key: str) -> tuple[str, str]:
    parts = key.strip().lower().split("_")
    if len(parts) != 2 or parts[0] not in ORDER_DIR or parts[1] not in ORDER_GRIP:
        raise ValueError(f"Bad --group {key!r}; use e.g. front_horizontal")
    return parts[0], parts[1]


def _collect_group_entries(
    env,
    disp_dir: str,
    orient: str,
    *,
    grid_side: int,
    head_hand_gap_min: float,
) -> list[dict[str, Any]]:
    os.environ["MOBILE_POSE_HEAD_HAND_X_GAP"] = str(head_hand_gap_min)
    bank = _get_pose_bank_entries()
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in bank:
        src_dir = str(e.get("dir"))
        o = str(e.get("orient"))
        d = next((k for k, v in SOURCE_DIR_FOR_DISPLAY.items() if v == src_dir), None)
        if d not in ORDER_DIR or o not in ORDER_GRIP:
            continue
        q = _default_qpos(env)
        q[QI_R_ARM] = [float(v) for v in e["r_arm_rad"]]
        env.sim.data.qpos[:] = q
        env.sim.forward()
        if not _right_hand_forward_vs_torso_head_ok(env.sim):
            continue
        buckets[(d, o)].append(e)
    return _select_xy_grid_balanced(buckets.get((disp_dir, orient), []), grid_side=grid_side)


def _render_tile(env, entry: dict[str, Any], *, tile_idx: int) -> Image.Image:
    q = _default_qpos(env)
    q[QI_R_ARM] = [float(v) for v in entry["r_arm_rad"]]
    im = _capture(env, q).convert("RGB")
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 470, 52], fill=(0, 0, 0))
    label_xyz = (
        f"x{int(round(float(entry.get('x', 50))))} "
        f"y{int(round(float(entry.get('y', 50))))} "
        f"z{int(round(float(entry.get('z', 50))))}"
    )
    jd = entry.get("joint_deg")
    if isinstance(jd, list) and len(jd) == 6:
        label_joint = "j=[" + ",".join(str(int(round(float(v)))) for v in jd) + "]"
    else:
        label_joint = "j=[?]"
    dr.text((8, 6), label_xyz, fill="white")
    dr.text((8, 28), label_joint, fill="white")
    return im


def _draw_plausible_box(im: Image.Image, *, plausible: bool) -> Image.Image:
    out = im.copy()
    dr = ImageDraw.Draw(out)
    w, h = out.size
    if plausible:
        dr.rectangle([2, 2, w - 3, h - 3], outline=(34, 197, 94), width=5)
        dr.rectangle([w - 118, 58, w - 8, 88], fill=(34, 197, 94))
        dr.text((w - 112, 64), "OK", fill="white")
    else:
        dr.rectangle([2, 2, w - 3, h - 3], outline=(180, 180, 180), width=2)
    return out


def _fill_prompt(template: str, entry: dict[str, Any], *, disp_dir: str, orient: str) -> str:
    jd = entry.get("joint_deg") or []
    jd_s = ",".join(str(int(round(float(v)))) for v in jd) if jd else "?"
    return (
        template.replace("{{DIR}}", disp_dir)
        .replace("{{ORIENT}}", orient)
        .replace("{{JOINT_DEG}}", jd_s)
        .replace("{{X}}", str(entry.get("x", "?")))
        .replace("{{Y}}", str(entry.get("y", "?")))
        .replace("{{Z}}", str(entry.get("z", "?")))
    )


def _compose_montage(
    tiles: list[Image.Image],
    *,
    title: str,
    grid_cols: int,
    out_path: Path,
) -> None:
    if not tiles:
        return
    cw, ch = tiles[0].size
    cols = grid_cols
    grid_rows = math.ceil(len(tiles) / cols)
    pad = 6
    header = 60
    canvas = Image.new(
        "RGB",
        (pad + cols * (cw + pad), header + pad + grid_rows * (ch + pad)),
        (248, 248, 248),
    )
    dr = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        ft = ImageFont.load_default()
    dr.text((10, 10), title, fill=(20, 20, 20), font=ft)
    for i, im in enumerate(tiles):
        rr, cc = divmod(i, cols)
        x0 = pad + cc * (cw + pad)
        y0 = header + pad + rr * (ch + pad)
        canvas.paste(im, (x0, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _load_resume(out_json: Path) -> dict[int, dict[str, Any]]:
    if not out_json.is_file():
        return {}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return {int(r["tile_idx"]): r for r in (data.get("results") or []) if "tile_idx" in r}


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_html(payload: dict[str, Any], out_path: Path) -> None:
    rows = payload.get("results") or []
    n_ok = sum(1 for r in rows if r.get("is_plausible") is True)
    cards = []
    for r in rows:
        ok = r.get("is_plausible")
        badge = (
            "<span class='ok'>plausible</span>"
            if ok
            else "<span class='no'>implausible</span>"
        )
        img = r.get("boxed_image", "")
        cards.append(
            f"<article><h3>#{r.get('tile_idx')} {badge}</h3>"
            f"<p>{html.escape(str(r.get('reason', '')))}</p>"
            f"<p class='meta'>model={html.escape(str(r.get('model', '')))} "
            f"joint={html.escape(str(r.get('joint_deg', '')))}</p>"
            f'<img src="{html.escape(img)}" loading="lazy"/></article>'
        )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pose plausibility — {html.escape(payload.get('group', ''))}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}
.wrap{{padding:16px}}h1{{font-size:1.2rem}}
.summary{{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
article{{background:#111827;border:1px solid #334155;border-radius:10px;padding:10px}}
img{{width:100%;height:auto;border-radius:6px}}
.ok{{color:#4ade80;font-weight:700}}.no{{color:#f87171;font-weight:700}}
.meta{{font-size:12px;color:#94a3b8}}
</style></head><body><div class="wrap">
<h1>Pose plausibility — {html.escape(payload.get('group', ''))}</h1>
<div class="summary">
<p>{n_ok}/{len(rows)} plausible · model={html.escape(str(payload.get('model', '')))}</p>
<p><a href="{html.escape(payload.get('montage_plausible_only') or '')}">plausible-only grid (green boxes)</a> ·
<a href="{html.escape(payload.get('montage_boxed', ''))}">all tiles</a></p>
</div>
<section class="grid">{''.join(cards)}</section></div></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def _finalize_outputs(
    *,
    args: argparse.Namespace,
    vis_dir: Path,
    tiles_dir: Path,
    disp_dir: str,
    orient: str,
    model_used: str,
    results: list[dict[str, Any]],
    boxed_tiles: list[Image.Image],
    grid_side: int,
) -> dict[str, Any]:
    montage_path = vis_dir / f"montage_{args.group}_boxed.png"
    _compose_montage(
        boxed_tiles,
        title=f"all tiles | {args.group} | green box = plausible",
        grid_cols=grid_side,
        out_path=montage_path,
    )

    plausible_only = [
        _draw_plausible_box(
            Image.open(tiles_dir / f"tile_{r['tile_idx']:02d}.png").convert("RGB"),
            plausible=True,
        )
        for r in results
        if r.get("is_plausible") is True
    ]
    montage_ok_path = vis_dir / f"montage_{args.group}_plausible_only.png"
    if plausible_only:
        ok_cols = min(grid_side, max(1, len(plausible_only)))
        _compose_montage(
            plausible_only,
            title=f"plausible only ({len(plausible_only)}) | {args.group}",
            grid_cols=ok_cols,
            out_path=montage_ok_path,
        )
    else:
        montage_ok_path = None

    out_json = Path(args.out_json)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "group": args.group,
        "direction_group": disp_dir,
        "gripper_orientation": orient,
        "model": model_used,
        "n_tiles": len(results),
        "n_plausible": sum(1 for r in results if r.get("is_plausible") is True),
        "prompt": str(Path(args.prompt).relative_to(_REPO)),
        "montage_boxed": str(montage_path.relative_to(_REPO)),
        "montage_plausible_only": (
            str(montage_ok_path.relative_to(_REPO)) if montage_ok_path else None
        ),
        "results": results,
    }
    _save_payload(out_json, payload)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {montage_path}", flush=True)

    html_path = vis_dir / f"review_{args.group}.html"
    html_payload = dict(payload)
    html_payload["montage_boxed"] = montage_path.name
    html_payload["montage_plausible_only"] = montage_ok_path.name if montage_ok_path else ""
    html_rows = []
    for r in results:
        hr = dict(r)
        hr["boxed_image"] = f"tiles/{Path(str(r.get('boxed_image', ''))).name}"
        html_rows.append(hr)
    html_payload["results"] = html_rows
    _write_html(html_payload, html_path)
    print(f"Wrote {html_path}", flush=True)

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)

    return payload


def run(args: argparse.Namespace) -> None:
    disp_dir, orient = _parse_group_key(args.group)
    grid_side = args.grid_cols
    vis_dir = Path(args.vis_dir)
    tiles_dir = vis_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    template = Path(args.prompt).read_text(encoding="utf-8")
    client = None if args.dry_run else _gemini_client()

    env = _make_env()
    out_json = Path(args.out_json)
    resume = _load_resume(out_json) if args.resume else {}
    results: list[dict[str, Any]] = []
    boxed_tiles: list[Image.Image] = []
    model_used = args.model

    try:
        entries = _collect_group_entries(
            env,
            disp_dir,
            orient,
            grid_side=grid_side,
            head_hand_gap_min=args.head_hand_gap_min,
        )
        print(f"[group] {disp_dir}+{orient}: {len(entries)} tiles", flush=True)

        for i, entry in enumerate(entries, start=1):
            tile_name = f"tile_{i:02d}.png"
            tile_path = tiles_dir / tile_name

            if args.resume and i in resume and resume[i].get("is_plausible") is not None:
                rec = dict(resume[i])
                im = Image.open(tile_path).convert("RGB") if tile_path.is_file() else _render_tile(env, entry, tile_idx=i)
                plausible = bool(rec.get("is_plausible"))
                boxed = Image.open(tiles_dir / f"tile_{i:02d}_boxed.png").convert("RGB") if (tiles_dir / f"tile_{i:02d}_boxed.png").is_file() else _draw_plausible_box(im, plausible=plausible)
                print(f"  [{i:02d}] resume {'OK' if plausible else 'NO'}", flush=True)
                boxed_tiles.append(boxed)
                results.append(rec)
                model_used = str(rec.get("model") or model_used)
                continue

            im = _render_tile(env, entry, tile_idx=i)
            im.save(tile_path)

            rec: dict[str, Any] = {
                "tile_idx": i,
                "direction_group": disp_dir,
                "gripper_orientation": orient,
                "x": entry.get("x"),
                "y": entry.get("y"),
                "z": entry.get("z"),
                "joint_deg": entry.get("joint_deg"),
                "r_arm_rad": entry.get("r_arm_rad"),
                "tile_image": str(tile_path.relative_to(_REPO)),
            }

            try:
                if args.dry_run:
                    rec["dry_run"] = True
                    rec["is_plausible"] = None
                    boxed = _draw_plausible_box(im, plausible=False)
                else:
                    prompt = _fill_prompt(template, entry, disp_dir=disp_dir, orient=orient)
                    text, model_used = _generate_with_fallback(
                        client,
                        model=args.model,
                        img_bytes=tile_path.read_bytes(),
                        prompt=prompt,
                    )
                    parsed = _extract_json(text)
                    plausible = bool(parsed.get("is_plausible"))
                    rec.update(
                        {
                            "is_plausible": plausible,
                            "confidence": parsed.get("confidence"),
                            "reason": parsed.get("reason"),
                            "model": model_used,
                            "raw": parsed,
                        }
                    )
                    boxed = _draw_plausible_box(im, plausible=plausible)
                    print(
                        f"  [{i:02d}] {'OK' if plausible else 'NO'} — {str(rec.get('reason', ''))[:70]}",
                        flush=True,
                    )
            except RuntimeError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[stop] API quota — saved {len(results)} results; rerun with --resume", flush=True)
                    break
                raise

            boxed_name = f"tile_{i:02d}_boxed.png"
            boxed_path = tiles_dir / boxed_name
            boxed.save(boxed_path)
            rec["boxed_image"] = str(boxed_path.relative_to(vis_dir))
            boxed_tiles.append(boxed)
            results.append(rec)

            _save_payload(
                out_json,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "group": args.group,
                    "model": model_used,
                    "n_tiles": len(results),
                    "n_plausible": sum(1 for r in results if r.get("is_plausible") is True),
                    "results": results,
                },
            )
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    _finalize_outputs(
        args=args,
        vis_dir=vis_dir,
        tiles_dir=tiles_dir,
        disp_dir=disp_dir,
        orient=orient,
        model_used=model_used,
        results=results,
        boxed_tiles=boxed_tiles,
        grid_side=grid_side,
    )

def main() -> None:
    p = argparse.ArgumentParser(description="Gemini per-tile pose plausibility (one pose group)")
    p.add_argument("--group", default=DEFAULT_GROUP, help="e.g. front_horizontal")
    p.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    p.add_argument("--head-hand-gap-min", type=float, default=DEFAULT_HEAD_HAND_GAP_MIN_M)
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    p.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
