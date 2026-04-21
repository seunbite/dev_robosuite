from __future__ import annotations

import copy
import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
OUT_ROOT = SEED / "q4_generic_recoil_hold_experiment"
RENDER_ROOT = MOTIONS / "q4_generic_recoil_hold_experiment" / "IIWA" / "IIWA"

ICONIC_SRC = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_SRC = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
HTML_OUT = OUT_ROOT / "prompt19_sophisticated_q4_generic_recoil_hold_compare_20260404_ko.html"
ICONIC_OUT = OUT_ROOT / "prompt19_sophisticated_q4_generic_recoil_hold_iconic_5cue.json"
CONTEXTUAL_OUT = OUT_ROOT / "prompt19_sophisticated_q4_generic_recoil_hold_contextual_5cue.json"
MANIFEST_OUT = OUT_ROOT / "manifest.json"


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _find_any_gif(base: Path, cue: str) -> Path | None:
    single = sorted(base.rglob(f"*_{cue}_p*.gif"))
    if single:
        return single[-1]
    tiled = sorted(base.rglob(f"*_{cue}_c*_tiled.gif"))
    if tiled:
        return tiled[-1]
    any_match = sorted(base.rglob(f"*_{cue}_*.gif"))
    return any_match[-1] if any_match else None


def _orig_motion_dir(dataset: str) -> Path:
    if dataset == "iconic":
        return MOTIONS / "v19_sophisticated" / "IIWA"
    return MOTIONS / "v19_sophisticated_contextual" / "IIWA"


def _update_stop_palm_out(row: dict) -> dict:
    row["movements"] = [row["movements"][0], row["movements"][1], row["movements"][2], row["movements"][3]]
    row["movements"][1]["parameters"]["directions"][0]["degrees"]["x"] = -8
    row["movements"][1]["parameters"]["directions"][0]["speed"] = 2.0
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["x"] = 28
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 3.3
    row["movements"][3]["parameters"]["repetition"] = 2
    row["movements"][3]["parameters"]["directions"] = [
        {"degrees": {"z": -14}, "speed": 2.3, "hold_time": 0.18},
        {"degrees": {"z": 4}, "speed": 2.0, "hold_time": 0.35},
    ]
    return row


def _update_cover_mouth_gasp(row: dict) -> dict:
    row["movements"] = [row["movements"][0], row["movements"][2], row["movements"][3]]
    row["movements"][1]["parameters"]["speed"] = 3.8
    row["movements"][1]["parameters"]["hold_time"] = 0.15
    row["movements"][2]["parameters"]["repetition"] = 3
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["y"] = 8
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 3.4
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 0.12
    row["movements"][2]["parameters"]["directions"][1]["degrees"]["y"] = -8
    row["movements"][2]["parameters"]["directions"][1]["speed"] = 3.4
    row["movements"][2]["parameters"]["directions"][1]["hold_time"] = 0.12
    return row


def _update_cheers_toast(row: dict) -> dict:
    row["movements"] = [row["movements"][0], row["movements"][2], row["movements"][3]]
    row["movements"][1]["parameters"]["radius"] = 45
    row["movements"][1]["parameters"]["speed"] = 2.0
    row["movements"][2]["parameters"]["repetition"] = 2
    row["movements"][2]["parameters"]["directions"] = [
        {"degrees": {"x": 12}, "speed": 2.8, "hold_time": 0.08},
        {"degrees": {"x": -4}, "speed": 2.3, "hold_time": 0.22},
    ]
    return row


def _update_slow_down_request(row: dict) -> dict:
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 0.55
    row["movements"][3]["parameters"]["directions"][0]["degrees"]["z"] = 14
    row["movements"][3]["parameters"]["directions"][0]["speed"] = 1.1
    row["movements"][3]["parameters"]["directions"][0]["hold_time"] = 0.05
    row["movements"][4]["parameters"]["directions"][0]["degrees"]["z"] = -16
    row["movements"][4]["parameters"]["directions"][0]["hold_time"] = 0.75
    return row


def _update_commit_action_fast_reach(row: dict) -> dict:
    row["movements"] = [row["movements"][0], row["movements"][2], row["movements"][3]]
    row["movements"][1]["parameters"]["distance"] = 40
    row["movements"][1]["parameters"]["speed"] = 4.3
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["x"] = 8
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 2.4
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 0.55
    return row


EXPERIMENTS = [
    {
        "dataset": "iconic",
        "idx": 16,
        "cue": "stop_palm_out",
        "confusable_with": ["protective_arm_out", "slow_down_request_palm_down"],
        "feature": "barrier lock with a palm-out pulse",
        "reasoning": "# Q1: A strong stop cue should read as a barrier appearing in front of the body, not just a forceful push.\n# Q2: candidates=P1) front+vertical+x55,y50,z55, P2) up+horizontal+x50,y50,z70, P3) down+vertical+x48,y50,z45; winner=P1 because the chest-level vertical hand is the cleanest barrier baseline.\n# Q3: options=M1 P1>movement(recoil)>movement(push), M2 P1>movement(push)>movement(wrist barrier pulse), M3 P1>path(line x)>hold; winner=M2 because the stop cue becomes more specific when the forward push ends in a visible palm-out barrier pulse instead of only relying on anticipation.\n# Q4: confusable_with=[protective_arm_out, slow_down_request_palm_down]; discriminative_feature=barrier-like palm-out pulse at the end of the push; options=C1 preparatory recoil before the push, C2 palm-out barrier pulse during the held stop, C3 final settle after the push; winner=C2 because the barrier pulse adds cue-specific evidence that this is an explicit command to stop rather than a generic defensive extension or calming press.",
        "mutate": _update_stop_palm_out,
    },
    {
        "dataset": "iconic",
        "idx": 24,
        "cue": "cover_mouth_gasp",
        "confusable_with": ["rub_eye_tired", "facepalm"],
        "feature": "post-arrival mouth-cover tremble",
        "reasoning": "# Q1: A gasp should read as a sudden cover of the mouth followed by a brief shocked tremble, not just a recoil plus reach.\n# Q2: candidates=P1) down+vertical, P2) front+horizontal; winner=P1 because the low start gives the hand enough travel to read as a startled reaction.\n# Q3: options=M1 pose>recoil>pose>tremble, M2 pose>pose>tremble, M3 pose>combined recoil+lift; winner=M2 because the direct rise to the mouth plus an immediate tremble reads more specifically as shock than a generic preparatory recoil.\n# Q4: confusable_with=[rub_eye_tired, facepalm]; discriminative_feature=small shock tremble after the hand reaches the mouth; options=C1 recoil before hand moves, C2 tremble after hand arrives, C3 final static hold at the mouth; winner=C2 because the mouth-cover tremble is the most cue-specific sign of shock and separates the cue from tired face-touching or frustrated facepalm motions.",
        "mutate": _update_cover_mouth_gasp,
    },
    {
        "dataset": "iconic",
        "idx": 25,
        "cue": "cheers_toast",
        "confusable_with": ["raising_hand_greeting", "firm_accept_forward_reach"],
        "feature": "distinct clink punctuation",
        "reasoning": "# Q1: A toast should read like lifting a glass into the air and punctuating the apex with a celebratory clink.\n# Q2: candidates=P1) front+horizontal+x55,y50,z45, P2) down+vertical+x48,y50,z35; winner=P1 because the frontal waist-level pose best resembles holding a drink before the lift.\n# Q3: options=M1 pose>movement(prep dip)>path(arc lift)>movement(clink), M2 pose>path(arc lift)>movement(clink), M3 pose>path(line forward); winner=M2 because the clink punctuation matters more than a generic anticipatory dip.\n# Q4: confusable_with=[raising_hand_greeting, firm_accept_forward_reach]; discriminative_feature=brief clink accent at the top of the toast; options=C1 preparatory dip before the lift, C2 sharper clink punctuation at the apex, C3 final static hold after the clink; winner=C2 because the clink is the strongest cue-specific evidence of a toast and better separates it from a greeting lift or forward offering motion.",
        "mutate": _update_cheers_toast,
    },
    {
        "dataset": "iconic",
        "idx": 45,
        "cue": "slow_down_request_palm_down",
        "confusable_with": ["stop_palm_out", "request_turn"],
        "feature": "rhythmic press-hold cadence",
        "reasoning": "# Q1: A slow-down request should feel like repeated calming downward presses, not a stop barrier or a social offering sweep.\n# Q2: candidates=P1) front+horizontal+x65,z55, P2) up+horizontal+x50,z70; winner=P1 because the chest-level palm-down pose is non-threatening and ideal for a calming press.\n# Q3: options=M1 pose>movement(repeat z), M2 pose(neutral)>pose(active)>movement(z down)>movement(z up)>movement(z down), M3 pose>movement(z down hold)>movement(z down hold); winner=M3 because the rhythm of repeated downward press-holds is more cue-specific than a generic single push with setup.\n# Q4: confusable_with=[stop_palm_out, request_turn]; discriminative_feature=gentle repeated bottom holds that create a calming cadence; options=C1 hold at the bottom of each press, C2 final sag after the presses, C3 sharper rebound between presses; winner=C1 because the repeated bottom holds make the cue read as slowing and calming rather than stopping or redirecting.",
        "mutate": _update_slow_down_request,
    },
    {
        "dataset": "contextual",
        "idx": 39,
        "cue": "commit_action_fast_reach",
        "confusable_with": ["firm_accept_forward_reach", "hesitation_pause_hold"],
        "feature": "terminal commitment lock after the thrust",
        "reasoning": "# Q1: A committed action should read as a decisive thrust that lands and locks, not just a recoil-plus-reach.\n# Q2: candidates=P1) front+horizontal+x50,y50,z50, P2) down+vertical+x45,y50,z35; winner=P1 because the chest-level start makes the forward commitment easiest to read.\n# Q3: options=M1 pose>movement(recoil)>path(thrust), M2 pose>path(thrust)>movement(lock), M3 pose>path(arc); winner=M2 because the landing lock is more discriminative of commitment than a generic preparatory recoil.\n# Q4: confusable_with=[firm_accept_forward_reach, hesitation_pause_hold]; discriminative_feature=terminal lock that makes the thrust feel fully committed; options=C1 preparatory recoil before the thrust, C2 final commitment lock after the thrust, C3 trailing settle after the thrust; winner=C2 because the terminal lock best separates committed action from a polite reach or an aborted motion.",
        "mutate": _update_commit_action_fast_reach,
    },
]


def _write_compare_html(rows: list[dict]) -> None:
    cards = []
    for row in rows:
        before_uri = Path(row["before_gif"]).resolve().as_uri() if row["before_gif"] else ""
        after_uri = Path(row["after_gif"]).resolve().as_uri() if row["after_gif"] else ""
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="meta"><span class="dataset {html.escape(row['dataset'])}">{html.escape(row['dataset'])}</span><span>c{row['idx']}</span></div>
                <h2>{html.escape(row['cue'])}</h2>
                <div class="sub">confusable_with: {html.escape(', '.join(row['confusable_with']))}</div>
                <div class="sub">discriminative_feature: <strong>{html.escape(row['discriminative_feature'])}</strong></div>
              </div>
              <div class="media-grid">
                <div class="media-card">
                  <div class="label">Before</div>
                  {f'<img src="{before_uri}" alt="before {html.escape(row["cue"])}">' if before_uri else '<div class="missing">missing</div>'}
                </div>
                <div class="media-card">
                  <div class="label">After Semantic Q4</div>
                  {f'<img src="{after_uri}" alt="after {html.escape(row["cue"])}">' if after_uri else '<div class="missing">render pending</div>'}
                </div>
              </div>
              <div class="text-block">
                <div class="label">Reasoning</div>
                <pre>{html.escape(row['reasoning'])}</pre>
              </div>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt 19 Sophisticated Q4 Recoil Hold Compare</title>
  <style>
    body {{ margin: 0; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background: #fff; color: #111; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p.lead {{ margin: 0 0 18px; color: #5d6670; }}
    .grid {{ display: grid; gap: 18px; }}
    .card {{ border: 1px solid #dde3e8; background: #fff; }}
    .hdr {{ padding: 14px 16px; border-bottom: 1px solid #eef2f5; }}
    .meta {{ display: flex; gap: 8px; font-size: 12px; color: #5d6670; margin-bottom: 6px; }}
    .dataset {{ padding: 2px 8px; border: 1px solid #d9e0e6; border-radius: 999px; }}
    .dataset.iconic {{ background: #e9f3ff; }}
    .dataset.contextual {{ background: #ebf8eb; }}
    .hdr h2 {{ margin: 0 0 6px; font-size: 20px; }}
    .sub {{ font-size: 13px; color: #5d6670; margin-top: 2px; }}
    .media-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 14px 16px; }}
    .media-card {{ border: 1px solid #eef2f5; padding: 8px; background: #fff; }}
    .media-card img {{ width: 100%; display: block; }}
    .label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #5d6670; margin-bottom: 6px; }}
    .text-block {{ padding: 0 16px 16px; }}
    .text-block pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f9fb; border: 1px solid #eef2f5; padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
    .missing {{ min-height: 180px; display: grid; place-items: center; background: #f7f9fb; color: #6c7680; }}
    @media (max-width: 960px) {{ .media-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated: Generic Recoil / Hold Q4 Retest</h1>
    <p class="lead">원래 Q4가 recoil이나 hold 쪽으로 수렴하던 항목들만 따로 뽑아, semantic accent 쪽으로 다시 설계한 5개 비교입니다.</p>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    iconic_src = {int(r["idx"]): r for r in _load_rows(ICONIC_SRC)}
    contextual_src = {int(r["idx"]): r for r in _load_rows(CONTEXTUAL_SRC)}
    iconic_rows = []
    contextual_rows = []
    summary_rows = []
    for spec in EXPERIMENTS:
        src = iconic_src if spec["dataset"] == "iconic" else contextual_src
        row = copy.deepcopy(src[spec["idx"]])
        row["reasoning"] = spec["reasoning"]
        row = spec["mutate"](row)
        row["recoil_hold_retest"] = {
            "confusable_with": spec["confusable_with"],
            "discriminative_feature": spec["feature"],
        }
        if spec["dataset"] == "iconic":
            iconic_rows.append(row)
        else:
            contextual_rows.append(row)
        summary_rows.append(
            {
                "dataset": spec["dataset"],
                "idx": spec["idx"],
                "cue": spec["cue"],
                "confusable_with": spec["confusable_with"],
                "discriminative_feature": spec["feature"],
                "reasoning": row["reasoning"],
                "before_gif": str(_find_any_gif(_orig_motion_dir(spec["dataset"]), spec["cue"]) or ""),
                "after_gif": str(_find_any_gif(RENDER_ROOT, spec["cue"]) or ""),
                "config": row,
            }
        )
    ICONIC_OUT.write_text(json.dumps(iconic_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    CONTEXTUAL_OUT.write_text(json.dumps(contextual_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_OUT.write_text(json.dumps({"count": len(summary_rows), "cues": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_compare_html(summary_rows)
    print("Wrote:", ICONIC_OUT)
    print("Wrote:", CONTEXTUAL_OUT)
    print("Wrote:", HTML_OUT)


if __name__ == "__main__":
    main()
