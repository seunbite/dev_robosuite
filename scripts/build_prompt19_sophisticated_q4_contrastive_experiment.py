from __future__ import annotations

import copy
import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
OUT_ROOT = SEED / "q4_contrastive_experiment"
RENDER_ROOT = MOTIONS / "q4_contrastive_experiment" / "IIWA" / "IIWA"

ICONIC_SRC = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_SRC = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
SHOT_SRC = SEED / "shot_configs_v19_sophisticated.json"

PROMPT_PATH = OUT_ROOT / "prompt19_sophisticated_q4_contrastive_prompt.md"
SHOT_OUT = OUT_ROOT / "shot_configs_v19_sophisticated_q4_contrastive.json"
ICONIC_OUT = OUT_ROOT / "prompt19_sophisticated_q4_contrastive_iconic_5cue.json"
CONTEXTUAL_OUT = OUT_ROOT / "prompt19_sophisticated_q4_contrastive_contextual_5cue.json"
HTML_OUT = OUT_ROOT / "prompt19_sophisticated_q4_contrastive_compare_20260404_ko.html"
MANIFEST_OUT = OUT_ROOT / "manifest.json"


PROMPT_TEXT = """# Prompt 19 Sophisticated: Q4 Contrastive Addendum

Use this addendum only for cues that already have Q4-style refinement.

## Goal
Q4 is not just for smoothing or naturalness. Its main purpose is to increase cue identifiability by separating the target cue from the most confusable neighboring cues.

## Required Q4 procedure
1. Identify the 2 most confusable cues for the target cue.
2. State the single discriminative feature that best separates the target from those confusable cues.
3. Propose exactly 3 distinct Q4 options with different roles:
   - C1: anticipatory beat
   - C2: cue-specific semantic accent
   - C3: follow-through, consequence, or posture modifier
4. Choose the option that increases recognizability the most, not just smoothness.

## Priority order
1. Stronger separation from confusable cues
2. Cue-specific semantic evidence
3. Naturalness

## Penalties
- Penalize generic recoil / lift / hold if it does not add cue-specific information.
- Do not choose a generic anticipation when a small semantic accent would make the cue easier to distinguish.
- Avoid endings that flatten the motion into a static pose unless the pause itself is the identifying feature.

## Output style for Q4
Include:
- confusable_with=[cue_a, cue_b]
- discriminative_feature=<short phrase>
- options=C1 ..., C2 ..., C3 ...
- winner=Cx because ...

The winner should explain why the selected beat distinguishes the target cue from the confusable alternatives.
"""


SHOT_REWRITES = {
    "Wave one hand side to side (Hello / Hi)": {
        "description": "The robot should not just raise the arm; it should clearly read as a greeting rather than yielding or presenting, so the wave accent must do the semantic work.",
        "planning_shot": "Confusable cues: Raise one hand briefly, Yield turn.\nDiscriminative feature: a visible wrist-side wave at the top, not a generic lift.\nQ4 rule: prefer the smallest peak accent that makes the motion unmistakably social.",
        "reasoning": "# Q1: A natural hello should read as a socially directed greeting, not just an arm raise.\n# Q2: candidates=P1) up+horizontal+x60,y50,z65, P2) front+horizontal+x70,y50,z60, P3) back+horizontal+x55,y50,z60; winner=P1 because the raised horizontal hand gives the clearest waving silhouette once the social accent is added.\n# Q3: options=M1 P1>movement(shoulder x fore-back notice)>movement(wrist y repeat), M2 P1>path(line y), M3 P2>movement(elbow y repeat); winner=M1 because the cue is strongest when the robot first seems to notice someone and then performs a visible wave.\n# Q4: confusable_with=[Raise one hand briefly, Yield turn]; discriminative_feature=peak wrist wave that reads as greeting rather than acknowledgment or redirection; options=C1 slight fore-back noticing beat before the wave, C2 stronger wrist wave at the peak, C3 final settle after the wave; winner=C2 because the peak wave is the smallest cue-specific accent that most clearly separates hello from a generic lift or polite redirect."
    },
    "Raise one hand briefly": {
        "description": "The robot should read as a brief acknowledgement, not a full greeting, by using a compact rise with a crisp top accent instead of a friendly repeated wave.",
        "planning_shot": "Confusable cues: Wave one hand side to side (Hello / Hi), Salute.\nDiscriminative feature: a brief upward acknowledgement with a short peak punctuation, not a repeated wave.",
        "reasoning": "# Q1: A brief acknowledgement should read as a short, contained raise rather than a greeting sequence.\n# Q2: candidates=P1) front+horizontal+x60,y50,z35, P2) up+horizontal+x55,y50,z55, P3) back+vertical+x45,y50,z35; winner=P1 because the low front baseline makes the short rise and quick top punctuation easiest to read.\n# Q3: options=M1 P1>movement(shoulder z dip)>path(line z up)>movement(wrist z accent), M2 P1>movement(shoulder z), M3 P2>pose>movement; winner=M1 because the brief dip-rise-accent pattern reads as acknowledgment without drifting into a greeting wave.\n# Q4: confusable_with=[Wave one hand side to side (Hello / Hi), Bring one hand to forehead in salute position (Salute)]; discriminative_feature=single crisp top accent instead of repeated wave or forehead lock; options=C1 preparatory dip before the lift, C2 single top accent at the peak, C3 trailing settle after the lift; winner=C2 because the top punctuation preserves the brevity of acknowledgement while avoiding the repeated social wave that would make it read as hello."
    },
    "Bring one hand to forehead in salute position (Salute)": {
        "description": "The robot should clearly enter the salute lane before locking the forehead angle so the motion cannot be confused with a casual hand-to-head gesture.",
        "planning_shot": "Confusable cues: Raise one hand briefly, Rotate finger near temple (Crazy).\nDiscriminative feature: crisp salute lane placement before the final forehead lock.",
        "reasoning": "# Q1: A salute should first establish the correct lateral lane and then sharply lock into the forehead angle.\n# Q2: candidates=P1) up+horizontal+x50,y50,z50, P2) right+vertical+x50,y100,z50, P3) up+vertical+x50,y75,z70; winner=P1 because it gives the clearest staging before the arm is placed laterally and folded into salute.\n# Q3: options=M1 P1>pose(right lane)>pose(forehead lane)>movement(elbow x+z fold), M2 P1>movement(shoulder+elbow), M3 P3>hold only; winner=M1 because the cue depends on staged lane placement before the final crisp angle.\n# Q4: confusable_with=[Raise one hand briefly, Rotate finger near temple (Crazy)]; discriminative_feature=explicit salute lane before the forehead lock; options=C1 lane placement before the fold, C2 release pose after the fold, C3 recoil after the fold; winner=C1 because the lane setup adds the most cue-specific information and prevents the gesture from collapsing into a generic hand-up or head-side motion."
    },
    "Rotate finger near temple (Crazy)": {
        "description": "The robot should first accuse outward, then reference the head, then make the circle, so the social meaning is clear rather than just a generic temple-side motion.",
        "planning_shot": "Confusable cues: Tap temple think, Salute.\nDiscriminative feature: outward accusation before the temple circle.",
        "reasoning": "# Q1: The crazy cue should read as a social accusation, not just a head-side circular motion.\n# Q2: candidates=P1) left+horizontal+x70,y40,z45, P2) front+horizontal+x58,y58,z60, P3) back+vertical+x45,y50,z60; winner=P2 because the front-facing start makes the outward accusation easiest to read before the motion shifts into the head-side sign.\n# Q3: options=M1 P2>movement(shoulder x point)>pose(P1 temple lane)>movement(wrist x tap)>path(arc xz), M2 P1>movement(wrist x+y jitter)>path(arc xz), M3 P2>path(arc xy); winner=M1 because the sequence becomes most legible when accusation, head reference, and symbolic circle appear in order.\n# Q4: confusable_with=[Tap temple think, Bring one hand to forehead in salute position (Salute)]; discriminative_feature=prior outward accusation; options=C1 point at you before the temple tap and circle, C2 recoil after the circle, C3 extra settle after the circle; winner=C1 because the accusation is the strongest cue-specific evidence that separates crazy from generic thinking or head-touch gestures."
    },
    "Rub one eye (Tired)": {
        "description": "The robot should specifically target the eye, rub locally, and then visibly sag so it reads as fatigue rather than a generic face touch.",
        "planning_shot": "Confusable cues: Cover mouth gasp, Scratch head confused.\nDiscriminative feature: eye-targeted rub followed by a fatigue sag.",
        "reasoning": "# Q1: Tiredness should read through eye-targeted contact, local rubbing, and an afterward loss of energy.\n# Q2: candidates=P1) left+vertical+x70,y55,z40, P2) front+horizontal+x55,y50,z65, P3) back+vertical+x45,y50,z55; winner=P1 because the lateral vertical hand best simulates direct eye contact before the tired drift and sagging drop.\n# Q3: options=M1 P1>movement(shoulder x+z slow drift)>movement(wrist y repeat)>path(line z down), M2 P1>path(arc yz), M3 P2>movement(elbow y repeat); winner=M1 because the cue is strongest when the approach, rub, and fatigue consequence all remain visible.\n# Q4: confusable_with=[Cover mouth gasp, Scratch head confused]; discriminative_feature=post-rub energy collapse; options=C1 sag downward after the rub, C2 final low linger pose, C3 recoil after the rub; winner=C1 because the downward collapse is the clearest cue-specific consequence of tiredness and better separates it from other face-touch gestures."
    },
    "Tilt head slightly (Curiosity)": {
        "description": "The robot should show a questioning lean rather than a generic pose change, so the small tilt must feel like inquisitive inspection.",
        "planning_shot": "Confusable cues: Listening lean substitute, Clarification request palm tilt.\nDiscriminative feature: small inquisitive tilt with a held questioning angle.",
        "reasoning": "# Q1: Curiosity should read as a contained, thoughtful head-side inquiry rather than a large directional motion.\n# Q2: candidates=P1) front+horizontal+x58,y50,z55, P2) right+vertical+x62,y48,z62; winner=P1 because the neutral front pose makes a subtle questioning tilt easier to read.\n# Q3: options=M1 P1>movement(shoulder y+x small tilt)>hold, M2 P1>path(arc), M3 P2>pose only; winner=M1 because a small staged tilt with a visible hold feels inquisitive without becoming a search or listen gesture.\n# Q4: confusable_with=[listening_lean_substitute, clarification_request_palm_tilt]; discriminative_feature=small held questioning angle; options=C1 slight prep breath before tilt, C2 tiny settle into the tilted angle, C3 final release back to neutral; winner=C2 because settling into the questioning angle keeps the cue readable as curiosity instead of a transient adjustment."
    },
    "Bow slightly from the waist (Respect / Greeting)": {
        "description": "The robot should read as respectful deference, not apology or thank-you, by making the forward lowering phase the most intentional part of the motion.",
        "planning_shot": "Confusable cues: Deep bow apology, Palms together thank you.\nDiscriminative feature: deliberate respectful dip without extra penitence or prayer-like emphasis.",
        "reasoning": "# Q1: Respectful greeting should read as a concise deferential dip, not a guilty collapse or thank-you pose.\n# Q2: candidates=P1) front+horizontal+x55,y50,z50, P2) down+vertical+x48,y50,z35; winner=P1 because the chest-level starting pose makes the respectful lowering and return easiest to read.\n# Q3: options=M1 P1>movement(shoulder x+z bow)>hold, M2 P1>path(arc), M3 P2>pose only; winner=M1 because the deliberate lowering and short held deferential angle communicate respect more clearly than a soft blended arc.\n# Q4: confusable_with=[Deep bow apology, Palms together thank you]; discriminative_feature=clean deferential dip without penitence; options=C1 preparatory breath before the bow, C2 short hold at the respectful low point, C3 sagging aftermath; winner=C2 because the low-point hold reinforces respect while avoiding the heavier emotional tone of apology."
    },
}


EXPERIMENTS = [
    {
        "dataset": "iconic",
        "idx": 1,
        "cue": "raising_hand_greeting",
        "confusable_with": ["wave_hi", "yield_turn"],
        "feature": "clear peak wave accent",
        "reasoning": "# Q1: A human greeting should read as a socially directed hello, not just a raised arm.\n# Q2: candidates=P1) down+vertical+x45,y50,z35, P2) front+horizontal+x60,y50,z45; winner=P1 because the low neutral start leaves room for a large lift and a visible wave accent.\n# Q3: options=M1 pose>path(line z), M2 pose>path(arc xz), M3 pose>path(arc xz)>movement(wrist y); winner=M3 because the curved lift plus a visible wave is more greeting-specific than a plain rise.\n# Q4: confusable_with=[wave_hi, yield_turn]; discriminative_feature=peak wrist wave that clearly signals greeting; options=C1 small anticipatory dip before the lift, C2 stronger repeated wrist wave at the apex, C3 trailing settle after the raise; winner=C2 because the apex wave is the smallest cue-specific accent that most clearly separates greeting from acknowledgment or redirection.",
        "mutate": lambda row: _update_raising_hand_greeting(row),
    },
    {
        "dataset": "iconic",
        "idx": 13,
        "cue": "point_self",
        "confusable_with": ["hand_over_heart", "gratitude_small_bow_hold"],
        "feature": "emphatic chest double-tap",
        "reasoning": "# Q1: A human points to themselves by drawing the hand inward and punctuating the chest target.\n# Q2: candidates=P1) front+horizontal+x60,y50,z50, P2) down+vertical+x48,y50,z45; winner=P1 because a forward chest-level baseline makes the inward self-reference easiest to read.\n# Q3: options=M1 pose>pose>movement(tap), M2 pose>path>hold; winner=M1 because staging the arrival at the chest before a separate tap reads more clearly as self-reference.\n# Q4: confusable_with=[hand_over_heart, gratitude_small_bow_hold]; discriminative_feature=distinct chest punctuation instead of a gentle sincerity hold; options=C1 preparatory lift before the inward move, C2 stronger double-tap on the chest, C3 final sincere hold on the chest; winner=C2 because the double-tap is the most cue-specific evidence of 'me' and best separates the cue from sincerity or gratitude gestures.",
        "mutate": lambda row: _update_point_self(row),
    },
    {
        "dataset": "iconic",
        "idx": 36,
        "cue": "visor_search",
        "confusable_with": ["salute", "listening_lean_substitute"],
        "feature": "forward peering extension during scan",
        "reasoning": "# Q1: A human searching the distance first establishes a visor at the brow and then leans into the scan.\n# Q2: candidates=P1) front+horizontal+x60,y50,z55, P2) up+vertical+x50,y50,z70; winner=P1 because the chest-level start gives a readable path into the brow visor lane.\n# Q3: options=M1 P1>path(line to brow)>movement(scan), M2 P1>pose(brow)>movement(scan), M3 P1>pose(brow)>path(x lean)>movement(scan); winner=M3 because the visor becomes most legible when the hand is placed crisply and the scan is accompanied by a forward peering extension.\n# Q4: confusable_with=[salute, listening_lean_substitute]; discriminative_feature=forward peering extension during the scan; options=C1 side-to-side scan only, C2 scan with a forward lean, C3 final hold at the brow; winner=C2 because the forward peering extension makes the motion read as searching the distance rather than saluting or simply cupping near the head.",
        "mutate": lambda row: _update_visor_search(row),
    },
    {
        "dataset": "contextual",
        "idx": 36,
        "cue": "prepare_action_raise_hold",
        "confusable_with": ["raise_index_one_moment", "commit_action_fast_reach"],
        "feature": "crisp lock at the ready peak",
        "reasoning": "# Q1: A human preparing to act raises the arm into position and visibly locks into readiness.\n# Q2: candidates=P1) down+vertical+x48,y50,z35, P2) front+horizontal+x55,y50,z55; winner=P1 because the low start makes the readying lift and held tension easiest to read.\n# Q3: options=M1 P1>path(z up)>hold, M2 P1>movement(z dip)>path(z up)>movement(elbow lock), M3 P1>path(z up)>movement(elbow lock stronger); winner=M3 because a stronger end lock distinguishes ready-to-act from a casual brief raise.\n# Q4: confusable_with=[raise_index_one_moment, commit_action_fast_reach]; discriminative_feature=crisp lock at the peak of the ready pose; options=C1 preparatory dip before the raise, C2 stronger elbow lock at the peak, C3 release after the hold; winner=C2 because the final lock is the most cue-specific evidence of readiness and best separates this cue from a simple acknowledgement or an immediate thrust.",
        "mutate": lambda row: _update_prepare_action_raise_hold(row),
    },
    {
        "dataset": "contextual",
        "idx": 38,
        "cue": "hesitation_pause_hold",
        "confusable_with": ["commit_action_fast_reach", "conditional_accept_pause_then_reach"],
        "feature": "sharp backward-downward loss of momentum",
        "reasoning": "# Q1: A human hesitation reads as an aborted start that suddenly loses confidence and freezes.\n# Q2: candidates=P1) front+horizontal+x60,y50,z50, P2) down+vertical+x48,y50,z45; winner=P1 because the neutral reach-ready pose makes the aborted start and recoil contrast easier to read.\n# Q3: options=M1 P1>path(short x)>movement(sharp x-z recoil/hold), M2 P1>path(short x)>movement(wobble)>hold, M3 P1>movement(z prep)>path(short x)>movement(sharp x-z recoil/hold); winner=M1 because the aborted advance followed by a coupled recoil is the clearest story of hesitation.\n# Q4: confusable_with=[commit_action_fast_reach, conditional_accept_pause_then_reach]; discriminative_feature=sharp backward-downward loss of momentum; options=C1 preparatory wobble before moving, C2 stronger coupled backward-and-downward recoil, C3 final static hold after the recoil; winner=C2 because the coupled recoil is the most discriminative evidence that the action was interrupted by doubt rather than merely delayed or completed.",
        "mutate": lambda row: _update_hesitation_pause_hold(row),
    },
]


def _update_raising_hand_greeting(row: dict) -> dict:
    row["movements"][2]["parameters"]["repetition"] = 3
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["y"] = 20
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 2.6
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 0.1
    row["movements"][2]["parameters"]["directions"][1]["degrees"]["y"] = -20
    row["movements"][2]["parameters"]["directions"][1]["speed"] = 2.6
    row["movements"][2]["parameters"]["directions"][1]["hold_time"] = 0.1
    return row


def _update_point_self(row: dict) -> dict:
    row["movements"][1]["parameters"]["pose"]["x"] = 40
    row["movements"][1]["parameters"]["pose"]["z"] = 58
    row["movements"][1]["parameters"]["speed"] = 2.1
    row["movements"][1]["parameters"]["hold_time"] = 0.08
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["x"] = 8
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 3.4
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 0.04
    row["movements"][2]["parameters"]["directions"][1]["degrees"]["x"] = -8
    row["movements"][2]["parameters"]["directions"][1]["speed"] = 3.4
    row["movements"][2]["parameters"]["directions"][1]["hold_time"] = 0.08
    return row


def _update_visor_search(row: dict) -> dict:
    row["movements"][1]["parameters"]["pose"]["x"] = 78
    row["movements"][1]["parameters"]["pose"]["z"] = 72
    row["movements"][1]["parameters"]["hold_time"] = 0.15
    row["movements"].insert(
        2,
        {
            "type": "path",
            "parameters": {
                "shape": "line",
                "joint": "shoulder",
                "axis": "x",
                "distance": 10,
                "speed": 1.0,
            },
        },
    )
    row["movements"][3]["parameters"]["directions"][0]["degrees"]["y"] = 18
    row["movements"][3]["parameters"]["directions"][0]["degrees"]["x"] = 6
    row["movements"][3]["parameters"]["directions"][1]["degrees"]["y"] = -36
    row["movements"][3]["parameters"]["directions"][1]["degrees"]["x"] = 6
    row["movements"][3]["parameters"]["directions"][0]["hold_time"] = 0.35
    row["movements"][3]["parameters"]["directions"][1]["hold_time"] = 0.35
    return row


def _update_prepare_action_raise_hold(row: dict) -> dict:
    row["movements"][2]["parameters"]["distance"] = 30
    row["movements"][2]["parameters"]["speed"] = 2.7
    row["movements"][3]["parameters"]["directions"][0]["degrees"]["x"] = 14
    row["movements"][3]["parameters"]["directions"][0]["degrees"]["z"] = 8
    row["movements"][3]["parameters"]["directions"][0]["speed"] = 3.4
    row["movements"][3]["parameters"]["directions"][0]["hold_time"] = 1.3
    return row


def _update_hesitation_pause_hold(row: dict) -> dict:
    row["movements"] = [row["movements"][0], row["movements"][2], row["movements"][3]]
    row["movements"][1]["parameters"]["distance"] = 12
    row["movements"][1]["parameters"]["speed"] = 0.85
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["x"] = -20
    row["movements"][2]["parameters"]["directions"][0]["degrees"]["z"] = -8
    row["movements"][2]["parameters"]["directions"][0]["speed"] = 3.0
    row["movements"][2]["parameters"]["directions"][0]["hold_time"] = 1.4
    return row


def _find_single_gif(base: Path, cue: str) -> Path | None:
    matches = sorted(base.rglob(f"*_{cue}_p*.gif"))
    return matches[-1] if matches else None


def _find_any_gif(base: Path, cue: str) -> Path | None:
    single = _find_single_gif(base, cue)
    if single:
        return single
    tiled = sorted(base.rglob(f"*_{cue}_c*_tiled.gif"))
    if tiled:
        return tiled[-1]
    any_match = sorted(base.rglob(f"*_{cue}_*.gif"))
    return any_match[-1] if any_match else None


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _build_shot_file() -> list[dict]:
    shots = copy.deepcopy(_load_rows(SHOT_SRC))
    for row in shots:
        patch = SHOT_REWRITES.get(row["cue"])
        if patch:
            row.update(patch)
    return shots


def _build_experiment_rows() -> tuple[list[dict], list[dict], list[dict]]:
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
        row["contrastive_experiment"] = {
            "confusable_with": spec["confusable_with"],
            "discriminative_feature": spec["feature"],
        }
        summary_rows.append(
            {
                "dataset": spec["dataset"],
                "idx": spec["idx"],
                "cue": spec["cue"],
                "confusable_with": spec["confusable_with"],
                "discriminative_feature": spec["feature"],
                "before_gif": str(_find_any_gif(_orig_motion_dir(spec["dataset"]), spec["cue"]) or ""),
                "after_gif": str(_find_any_gif(RENDER_ROOT, spec["cue"]) or ""),
                "reasoning": row["reasoning"],
                "config": row,
            }
        )
        if spec["dataset"] == "iconic":
            iconic_rows.append(row)
        else:
            contextual_rows.append(row)
    return iconic_rows, contextual_rows, summary_rows


def _orig_motion_dir(dataset: str) -> Path:
    if dataset == "iconic":
        return MOTIONS / "v19_sophisticated" / "IIWA"
    return MOTIONS / "v19_sophisticated_contextual" / "IIWA"


def _rel_uri(path: str | Path, html_path: Path) -> str:
    p = Path(path)
    if not p or not str(p):
        return ""
    return p.resolve().as_uri()


def _write_compare_html(rows: list[dict]) -> None:
    cards = []
    for row in rows:
        before_uri = _rel_uri(row["before_gif"], HTML_OUT) if row["before_gif"] else ""
        after_uri = _rel_uri(row["after_gif"], HTML_OUT) if row["after_gif"] else ""
        conf = ", ".join(row["confusable_with"])
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="meta"><span class="dataset {html.escape(row['dataset'])}">{html.escape(row['dataset'])}</span><span>c{row['idx']}</span></div>
                <h2>{html.escape(row['cue'])}</h2>
                <div class="sub">confusable_with: {html.escape(conf)}</div>
                <div class="sub">discriminative_feature: <strong>{html.escape(row['discriminative_feature'])}</strong></div>
              </div>
              <div class="media-grid">
                <div class="media-card">
                  <div class="label">Before</div>
                  {f'<img src="{before_uri}" alt="before {html.escape(row["cue"])}">' if before_uri else '<div class="missing">missing</div>'}
                </div>
                <div class="media-card">
                  <div class="label">After Contrastive Q4</div>
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
  <title>Prompt 19 Sophisticated Contrastive Q4 Compare</title>
  <style>
    body {{ margin: 0; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background: #fff; color: #111; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p.lead {{ margin: 0 0 18px; color: #5d6670; }}
    .paths {{ margin-bottom: 20px; font-size: 13px; color: #5d6670; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
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
    .media-card img {{ width: 100%; display: block; background: #fff; }}
    .label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #5d6670; margin-bottom: 6px; }}
    .text-block {{ padding: 0 16px 16px; }}
    .text-block pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f9fb; border: 1px solid #eef2f5; padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
    .missing {{ min-height: 180px; display: grid; place-items: center; background: #f7f9fb; color: #6c7680; }}
    @media (max-width: 960px) {{ .media-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated: Contrastive Q4 Pilot</h1>
    <p class="lead">기존 Q4보다 cue 식별도를 더 올리기 위해, confusable cue를 먼저 잡고 cue-specific accent를 우선하는 규칙으로 5개만 다시 설계한 파일입니다.</p>
    <div class="paths">prompt: {html.escape(str(PROMPT_PATH))}<br>shot: {html.escape(str(SHOT_OUT))}<br>iconic config: {html.escape(str(ICONIC_OUT))}<br>contextual config: {html.escape(str(CONTEXTUAL_OUT))}</div>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)

    PROMPT_PATH.write_text(PROMPT_TEXT, encoding="utf-8")

    shots = _build_shot_file()
    SHOT_OUT.write_text(json.dumps(shots, ensure_ascii=False, indent=2), encoding="utf-8")

    iconic_rows, contextual_rows, summary_rows = _build_experiment_rows()
    ICONIC_OUT.write_text(json.dumps(iconic_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    CONTEXTUAL_OUT.write_text(json.dumps(contextual_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "prompt_path": str(PROMPT_PATH),
        "shot_path": str(SHOT_OUT),
        "iconic_config_path": str(ICONIC_OUT),
        "contextual_config_path": str(CONTEXTUAL_OUT),
        "render_output_dir": str(RENDER_ROOT),
        "count": len(summary_rows),
        "cues": summary_rows,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_compare_html(summary_rows)
    print("Wrote:", PROMPT_PATH)
    print("Wrote:", SHOT_OUT)
    print("Wrote:", ICONIC_OUT)
    print("Wrote:", CONTEXTUAL_OUT)
    print("Wrote:", HTML_OUT)


if __name__ == "__main__":
    main()
