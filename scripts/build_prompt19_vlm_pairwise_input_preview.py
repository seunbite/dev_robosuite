from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import fire

from run_prompt19_vlm_pairwise_compare import (
    DATASET_CONFIGS,
    OUT_ROOT,
    _gif_to_mp4,
    _latest_single_gif,
    _load_row,
    _make_frame_strip_compare,
    _make_trajectory_compare,
)


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"


SAMPLES = [
    ("iconic", 50),
    ("contextual", 44),
]


def _build_preview_assets(dataset: str, cue_idx: int, out_dir: Path) -> dict:
    spec = DATASET_CONFIGS[dataset]
    sophisticated_row = _load_row(spec["sophisticated"], cue_idx)
    no_reasoning_row = _load_row(spec["no_reasoning"], cue_idx)
    cue = sophisticated_row["cue"]

    sophisticated_gif = _latest_single_gif(spec["sophisticated_gif_dir"], cue)
    no_reasoning_gif = _latest_single_gif(spec["no_reasoning_gif_dir"], cue)
    if sophisticated_gif is None or no_reasoning_gif is None:
        raise FileNotFoundError(f"Missing GIF for {dataset} c{cue_idx} {cue}")

    cue_dir = out_dir / f"{dataset}_c{cue_idx}_{cue}"
    cue_dir.mkdir(parents=True, exist_ok=True)

    mp4_a = _gif_to_mp4(sophisticated_gif, cue_dir / "A_sophisticated.mp4")
    mp4_b = _gif_to_mp4(no_reasoning_gif, cue_dir / "B_no_reasoning.mp4")
    strip = _make_frame_strip_compare(cue, sophisticated_gif, no_reasoning_gif, cue_dir / "compare_gif_strip.png", num_frames=6)
    traj = _make_trajectory_compare(
        cue=cue,
        cue_idx=cue_idx,
        left_row=sophisticated_row,
        right_row=no_reasoning_row,
        left_config=spec["sophisticated"],
        right_config=spec["no_reasoning"],
        out_path=cue_dir / "compare_trajectory.png",
        hz=8,
    )

    return {
        "dataset": dataset,
        "idx": cue_idx,
        "cue": cue,
        "mp4_a": mp4_a,
        "mp4_b": mp4_b,
        "strip": strip,
        "traj": traj,
        "gif_a": sophisticated_gif,
        "gif_b": no_reasoning_gif,
    }


def build(output_name: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_root = OUT_ROOT / f"input_preview_{stamp}"
    preview_root.mkdir(parents=True, exist_ok=True)

    samples = [_build_preview_assets(dataset, cue_idx, preview_root) for dataset, cue_idx in SAMPLES]

    cards = []
    for sample in samples:
        rel_mp4_a = sample["mp4_a"].relative_to(preview_root)
        rel_mp4_b = sample["mp4_b"].relative_to(preview_root)
        rel_strip = sample["strip"].relative_to(preview_root)
        rel_traj = sample["traj"].relative_to(preview_root)
        cards.append(
            f"""
            <section class="sample">
              <div class="sample-head">
                <h2>{html.escape(sample["dataset"])} c{sample["idx"]} {html.escape(sample["cue"])}</h2>
                <div class="meta">IIWA | A = Sophisticated | B = No Reasoning</div>
              </div>

              <div class="block">
                <div class="block-head">
                  <h3>1. MP4</h3>
                  <div class="desc">VLM에는 두 개의 별도 video가 들어갑니다. 첫 번째가 Motion A, 두 번째가 Motion B입니다.</div>
                </div>
                <div class="video-grid">
                  <div class="pane">
                    <div class="label">A Sophisticated</div>
                    <video controls loop muted playsinline src="{html.escape(str(rel_mp4_a))}"></video>
                  </div>
                  <div class="pane">
                    <div class="label">B No Reasoning</div>
                    <video controls loop muted playsinline src="{html.escape(str(rel_mp4_b))}"></video>
                  </div>
                </div>
              </div>

              <div class="block">
                <div class="block-head">
                  <h3>2. GIF Strip</h3>
                  <div class="desc">한 장의 비교 이미지입니다. 위 row가 Motion A, 아래 row가 Motion B이고, 각 row는 시간 순 프레임 strip입니다.</div>
                </div>
                <img class="wide" src="{html.escape(str(rel_strip))}" alt="gif strip preview">
              </div>

              <div class="block">
                <div class="block-head">
                  <h3>3. Last Frame + Trajectory</h3>
                  <div class="desc">한 장의 비교 이미지입니다. 왼쪽이 Motion A, 오른쪽이 Motion B이고, 마지막 프레임 위에 trajectory를 노랑에서 보라로 오버레이합니다.</div>
                </div>
                <img class="wide" src="{html.escape(str(rel_traj))}" alt="trajectory preview">
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt19 VLM Pairwise Input Preview</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#d9e0e7; --ink:#111; --muted:#66717d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width: 1800px; margin:0 auto; padding: 24px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin:0; font-size:24px; }}
    h3 {{ margin:0; font-size:18px; }}
    .lead {{ margin:0 0 10px; color:var(--muted); }}
    .sample {{ margin-top:30px; border-top:1px solid var(--line); padding-top:20px; }}
    .sample-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; margin-bottom:14px; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .block {{ border:1px solid var(--line); margin-top:16px; padding:14px; }}
    .block-head {{ margin-bottom:10px; }}
    .desc {{ margin-top:4px; color:var(--muted); font-size:13px; }}
    .video-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    .pane {{ min-width:0; }}
    .label {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    video, img.wide {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }}
    @media (max-width: 900px) {{ .video-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 VLM Pairwise Input Preview</h1>
    <p class="lead">VLM 비교 실험에서 실제로 들어가는 입력 형태를 미리 보는 페이지입니다. 기준은 IIWA, A= Sophisticated, B= No Reasoning입니다.</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""

    if output_name is None:
        output_name = "prompt19_vlm_pairwise_input_preview.html"
    out_path = preview_root / output_name
    out_path.write_text(page, encoding="utf-8")
    print(out_path)
    return str(out_path)


if __name__ == "__main__":
    fire.Fire({"build": build})
