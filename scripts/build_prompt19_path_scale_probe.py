from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"

OUT_DIR = SEED / "path_scale_probe"
CONFIG_PATH = OUT_DIR / "path_scale_probe_configs.json"
HTML_PATH = OUT_DIR / "path_scale_probe_20260404_ko.html"
RENDER_DIR = MOTIONS / "path_scale_probe" / "IIWA" / "IIWA"


ROWS = [
    {
        "idx": 0,
        "cue": "throat_cut_small_line_probe",
        "description": "Small lateral line intended to feel like a short throat-cut stroke. The goal is to see whether distance=10 already reads as a compact, local slash.",
        "state": "handmade_probe",
        "reasoning": "# Q1: A throat-cut cue should read as a short, local horizontal slash across the neck line.\n# Q2: candidates=P1 left+horizontal+x55,y35,z65, P2 front+horizontal+x60,y50,z60; winner=P1 because the lateral head-side start already places the hand near the neck lane.\n# Q3: options=M1 pose>path(line y short), M2 pose>movement(y), M3 pose>path(line y long); winner=M1 because a short clean path is the simplest way to test local slash scale.\n# Q4: options=C1 no extra beat, C2 recoil before slash, C3 sag after slash; winner=C1 because this probe is only testing path scale, not contextual reinforcement.",
        "movements": [
            {
                "type": "pose",
                "parameters": {
                    "pose": {
                        "dir": "left",
                        "x": 55,
                        "y": 35,
                        "z": 65,
                        "gripper_orientation": "horizontal",
                    },
                    "speed": 1.6,
                    "hold_time": 0.2,
                },
            },
            {
                "type": "path",
                "parameters": {
                    "shape": "line",
                    "joint": "shoulder",
                    "axis": "y",
                    "distance": 10,
                    "speed": 2.4,
                },
            },
        ],
    },
    {
        "idx": 1,
        "cue": "blackboard_big_line_probe",
        "description": "Large forward line intended to feel like drawing a big, extended stroke on a board in front of the robot. The goal is to see whether distance=120 reads like a clearly large spatial mark.",
        "state": "handmade_probe",
        "reasoning": "# Q1: A big board-writing stroke should read as a broad, extended line pushed out across space, not a local twitch.\n# Q2: candidates=P1 front+horizontal+x78,y28,z78, P2 up+horizontal+x60,y50,z85; winner=P1 because it places the hand in front of an imagined writing surface and leaves room for a long forward stroke.\n# Q3: options=M1 pose>path(line x long), M2 pose>path(arc xz), M3 pose>movement(x); winner=M1 because a long straight line is the cleanest probe for a board-sized stroke.\n# Q4: options=C1 no extra beat, C2 prep tap before stroke, C3 hold at the end; winner=C1 because this probe is focused on the size of the path itself.",
        "movements": [
            {
                "type": "pose",
                "parameters": {
                    "pose": {
                        "dir": "front",
                        "x": 78,
                        "y": 28,
                        "z": 78,
                        "gripper_orientation": "horizontal",
                    },
                    "speed": 1.2,
                    "hold_time": 0.2,
                },
            },
            {
                "type": "path",
                "parameters": {
                    "shape": "line",
                    "joint": "shoulder",
                    "axis": "x",
                    "distance": 120,
                    "speed": 1.6,
                },
            },
        ],
    },
    {
        "idx": 2,
        "cue": "crazy_small_arc_probe",
        "description": "Small temple-side arc intended to feel like a compact crazy-circle. The goal is to see whether radius=12 and sweep=360 read as a tight local loop rather than a broad flourish.",
        "state": "handmade_probe",
        "reasoning": "# Q1: A crazy sign near the temple should stay very local and feel like a compact circular scribble beside the head.\n# Q2: candidates=P1 left+vertical+x70,y40,z70, P2 front+horizontal+x58,y58,z60; winner=P1 because the side head lane keeps the arc anchored near the temple rather than floating in front of the torso.\n# Q3: options=M1 pose>path(arc xz small), M2 pose>movement(wrist z wobble), M3 pose>path(arc xz large); winner=M1 because a small circular path is the cleanest way to probe a compact temple-side arc.\n# Q4: options=C1 no extra beat, C2 accusatory point before the loop, C3 recoil after the loop; winner=C1 because this probe is only testing arc scale and sweep.",
        "movements": [
            {
                "type": "pose",
                "parameters": {
                    "pose": {
                        "dir": "left",
                        "x": 70,
                        "y": 40,
                        "z": 70,
                        "gripper_orientation": "vertical",
                    },
                    "speed": 1.4,
                    "hold_time": 0.1,
                },
            },
            {
                "type": "path",
                "parameters": {
                    "shape": "arc",
                    "joint": "wrist",
                    "plane": "xz",
                    "radius": 12,
                    "sweep": 360,
                    "speed": 1.2,
                    "direction": "cw",
                },
            },
        ],
    },
    {
        "idx": 3,
        "cue": "overhead_semicircle_arc_probe",
        "description": "Large overhead semicircle intended to feel like tracing a broad half-arc above the head. The goal is to see whether radius=25 and sweep=180 read as a clear overhead semicircle.",
        "state": "handmade_probe",
        "reasoning": "# Q1: An overhead semicircle should begin above the head and travel in one broad half-arc, not a tiny local loop.\n# Q2: candidates=P1 up+vertical+x50,y50,z100, P2 front+horizontal+x60,y50,z85; winner=P1 because starting from the apex above the head gives enough vertical space for a large visible semicircle.\n# Q3: options=M1 pose>path(arc yz semicircle), M2 pose>path(arc xz shallow), M3 pose>movement(z swing); winner=M1 because a single broad arc in the yz plane is the clearest probe for an overhead half-circle.\n# Q4: options=C1 no extra beat, C2 preparatory lift, C3 end hold; winner=C1 because this probe is focused on sweep size, not contextual embellishment.",
        "movements": [
            {
                "type": "pose",
                "parameters": {
                    "pose": {
                        "dir": "up",
                        "x": 50,
                        "y": 50,
                        "z": 100,
                        "gripper_orientation": "vertical",
                    },
                    "speed": 1.4,
                    "hold_time": 0.1,
                },
            },
            {
                "type": "path",
                "parameters": {
                    "shape": "arc",
                    "joint": "shoulder",
                    "plane": "yz",
                    "radius": 25,
                    "sweep": 180,
                    "speed": 1.2,
                    "direction": "ccw",
                },
            },
        ],
    },
]


def _find_gif(cue: str) -> Path | None:
    matches = sorted(RENDER_DIR.glob(f"*_{cue}_p*.gif"))
    if matches:
        return matches[-1]
    any_matches = sorted(RENDER_DIR.glob(f"*_{cue}_*.gif"))
    return any_matches[-1] if any_matches else None


def write_config() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(ROWS, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html() -> None:
    cards = []
    for row in ROWS:
        gif = _find_gif(row["cue"])
        gif_html = (
            f'<img src="{gif.resolve().as_uri()}" alt="{html.escape(row["cue"])}">'
            if gif
            else '<div class="missing">render pending</div>'
        )
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <h2>c{row['idx']} {html.escape(row['cue'])}</h2>
                <div class="sub">{html.escape(row['description'])}</div>
              </div>
              <div class="media">{gif_html}</div>
              <div class="body">
                <div class="label">Config</div>
                <pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>
              </div>
            </article>
            """
        )

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Path Scale Probe</title>
  <style>
    body {{ margin: 0; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background: #fff; color: #111; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .lead {{ margin: 0 0 18px; color: #5d6670; }}
    .grid {{ display: grid; gap: 18px; }}
    .card {{ border: 1px solid #dde3e8; background: #fff; }}
    .hdr {{ padding: 14px 16px; border-bottom: 1px solid #eef2f5; }}
    .hdr h2 {{ margin: 0 0 6px; font-size: 20px; }}
    .sub {{ color: #5d6670; font-size: 14px; line-height: 1.45; }}
    .media {{ padding: 14px 16px; }}
    .media img {{ width: 100%; display: block; }}
    .body {{ padding: 0 16px 16px; }}
    .label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; color: #5d6670; margin-bottom: 6px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f9fb; border: 1px solid #eef2f5; padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
    .missing {{ min-height: 220px; display: grid; place-items: center; background: #f7f9fb; color: #6c7680; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Path Scale Probe</h1>
    <p class="lead">프롬프트에 `distance/radius/sweep` 가이드를 넣기 전에, line과 arc의 작은/큰 예시가 실제로 얼마나 다르게 보이는지 보기 위한 최소 샘플입니다.</p>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    write_config()
    write_html()
    print(f"Wrote config: {CONFIG_PATH}")
    print(f"Wrote html: {HTML_PATH}")


if __name__ == "__main__":
    main()
