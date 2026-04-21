import html
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "adhoc" / "test" / "results"
ROBOTS = ["IIWA", "Panda", "XArm7"]
SETS = [
    {
        "slug": "sophisticated_iconic",
        "title": "Prompt19 Sophisticated + Iconic",
        "config_path": ROOT / "data" / "seed" / "prompt19_sophisticated_iconic.json",
        "motion_dir": ROOT / "data" / "motions" / "v19_sophisticated",
    },
    {
        "slug": "sophisticated_contextual",
        "title": "Prompt19 Sophisticated + Contextual",
        "config_path": ROOT / "data" / "seed" / "prompt19_sophisticated_contextual.json",
        "motion_dir": ROOT / "data" / "motions" / "v19_sophisticated_contextual",
    },
]


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _latest_tiled_gif(motion_root: Path, robot: str, cue: str, cue_idx: int) -> Path | None:
    safe_cue = _safe_name(cue)
    robot_dir = motion_root / robot
    if not robot_dir.exists():
        return None
    matches = sorted(
        robot_dir.glob(f"*_{safe_cue}_c{cue_idx}_tiled.gif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"prompt19_sophisticated_three_robot_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    dataset_summaries = []
    set_cards = []
    for spec in SETS:
        rows = json.loads(spec["config_path"].read_text())
        cards = []
        counts = {robot: 0 for robot in ROBOTS}
        for row in rows:
            cue_idx = int(row["idx"])
            cue = row["cue"]
            description = row.get("description", "")
            robot_media = []
            for robot in ROBOTS:
                gif_path = _latest_tiled_gif(spec["motion_dir"], robot, cue, cue_idx)
                if gif_path is not None:
                    counts[robot] += 1
                robot_media.append({"robot": robot, "gif_path": gif_path})
            cards.append(
                {
                    "idx": cue_idx,
                    "cue": cue,
                    "description": description,
                    "robot_media": robot_media,
                }
            )
        dataset_summaries.append({"title": spec["title"], "total": len(rows), "counts": counts})
        set_cards.append({"title": spec["title"], "slug": spec["slug"], "cards": cards})

    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Prompt19 Sophisticated Three-Robot Compare</title>",
        "<style>",
        ":root{--bg:#f3efe8;--paper:#fffdf8;--ink:#1e1d1a;--muted:#6a655c;--line:#d8d0c2;--accent:#8f5f2d;--soft:#f7f0e4;}",
        "*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Georgia,'Iowan Old Style',serif;}",
        ".wrap{max-width:1700px;margin:0 auto;padding:24px 18px 80px}",
        ".hero{background:linear-gradient(135deg,#f8f1e6,#efe4d1);border:1px solid var(--line);border-radius:24px;padding:24px 26px;margin-bottom:22px;box-shadow:0 10px 30px rgba(58,43,24,.06)}",
        ".hero h1{margin:0 0 8px;font-size:34px;line-height:1.1}",
        ".hero p{margin:0;color:var(--muted);font-size:16px;line-height:1.5;max-width:1050px}",
        ".summary{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}",
        ".chip{background:var(--paper);border:1px solid var(--line);border-radius:999px;padding:10px 14px;font:600 13px/1.2 -apple-system,BlinkMacSystemFont,sans-serif}",
        ".nav{position:sticky;top:0;z-index:20;background:rgba(243,239,232,.92);backdrop-filter:blur(8px);padding:10px 0 14px;margin-bottom:18px;border-bottom:1px solid rgba(216,208,194,.75)}",
        ".nav a{display:inline-block;margin-right:8px;margin-bottom:8px;padding:10px 14px;border-radius:999px;background:var(--paper);border:1px solid var(--line);color:var(--ink);text-decoration:none;font:600 13px/1.2 -apple-system,BlinkMacSystemFont,sans-serif}",
        ".set{margin-top:26px}",
        ".set h2{margin:0 0 14px;font-size:28px}",
        ".grid{display:grid;gap:18px}",
        ".card{background:var(--paper);border:1px solid var(--line);border-radius:22px;overflow:hidden;box-shadow:0 12px 28px rgba(51,36,18,.05)}",
        ".card-hd{padding:18px 20px 14px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#fffdfa,#f8f1e7)}",
        ".eyebrow{font:700 12px/1.2 -apple-system,BlinkMacSystemFont,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}",
        ".title{font-size:26px;line-height:1.15;margin:0 0 8px}",
        ".desc{margin:0;color:var(--muted);font-size:15px;line-height:1.55}",
        ".robots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0}",
        ".robot{padding:16px;border-right:1px solid var(--line)} .robot:last-child{border-right:none}",
        ".robot h3{margin:0 0 10px;font:700 16px/1.2 -apple-system,BlinkMacSystemFont,sans-serif}",
        ".media{background:#f4ede2;border:1px solid var(--line);border-radius:16px;overflow:hidden;min-height:320px;display:flex;align-items:center;justify-content:center}",
        ".media img{display:block;width:100%;height:auto;background:white}",
        ".missing{padding:24px;color:var(--muted);font:600 14px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;text-align:center}",
        ".meta{margin-top:10px;font:600 12px/1.4 -apple-system,BlinkMacSystemFont,sans-serif;color:var(--muted)}",
        ".meta a{color:var(--accent);text-decoration:none}",
        "@media (max-width:1200px){.robots{grid-template-columns:1fr}.robot{border-right:none;border-top:1px solid var(--line)}.robot:first-child{border-top:none}}",
        "</style>",
        "</head>",
        "<body>",
        "<div class='wrap'>",
        "<section class='hero'>",
        "<h1>Prompt19 Sophisticated Three-Robot Compare</h1>",
        "<p>Each cue shows the latest tiled GIF for <strong>IIWA</strong>, <strong>Panda</strong>, and <strong>XArm7</strong> side by side so we can compare the same motion prompt across robots. Missing renders are shown explicitly.</p>",
        "<div class='summary'>",
    ]

    for item in dataset_summaries:
        counts_text = " | ".join(f"{robot} {item['counts'][robot]}/{item['total']}" for robot in ROBOTS)
        parts.append(f"<div class='chip'>{_esc(item['title'])}: {_esc(counts_text)}</div>")

    parts.extend(["</div>", "</section>", "<div class='nav'>"])
    for spec in SETS:
        parts.append(f"<a href='#{_esc(spec['slug'])}'>{_esc(spec['title'])}</a>")
    parts.append("</div>")

    for section in set_cards:
        parts.append(f"<section class='set' id='{_esc(section['slug'])}'>")
        parts.append(f"<h2>{_esc(section['title'])}</h2>")
        parts.append("<div class='grid'>")
        for card in section["cards"]:
            parts.append("<article class='card'>")
            parts.append("<div class='card-hd'>")
            parts.append(f"<div class='eyebrow'>Cue {card['idx']}</div>")
            parts.append(f"<h3 class='title'>{_esc(card['cue'])}</h3>")
            parts.append(f"<p class='desc'>{_esc(card['description'])}</p>")
            parts.append("</div>")
            parts.append("<div class='robots'>")
            for robot_item in card["robot_media"]:
                gif_path = robot_item["gif_path"]
                parts.append("<section class='robot'>")
                parts.append(f"<h3>{_esc(robot_item['robot'])}</h3>")
                if gif_path is None:
                    parts.append("<div class='media'><div class='missing'>missing render</div></div>")
                else:
                    rel = _rel(gif_path, out_path.parent)
                    parts.append(f"<div class='media'><img src='{_esc(rel)}' alt='{_esc(robot_item['robot'])} {_esc(card['cue'])}'></div>")
                    parts.append(f"<div class='meta'><a href='{_esc(rel)}'>open gif</a></div>")
                parts.append("</section>")
            parts.append("</div>")
            parts.append("</article>")
        parts.append("</div>")
        parts.append("</section>")

    parts.extend(["</div>", "</body>", "</html>"])
    out_path.write_text("".join(parts), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
