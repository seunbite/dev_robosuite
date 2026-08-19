# Can LLMs turn gesture knowledge into recognizable robot motion?

## Yes, but approximately.

**Figure 1.** Human cue identification (yes/no)
- Graph: `data/results/evaluation/embodiment_cue_id/cueid_human_cue_identification.png`
- Finding: Yes—LLMs can produce approximate robot gestures—but humans recognize them far less often than human-authored motion, especially on non-humanoid platforms with a limited action space and fewer joints.
- A human labeler authored Motion Codes for 30 gesture cues on each embodiment; those codes were rendered to GIFs (human-authored).
- Separately, Gemini-2.5-Pro filled the same cue×embodiment slots; those codes were also rendered (Gemini-made).
- A human identifier then answered a yes/no question per GIF: if shown this rendered GIF, is it recognizable as that cue?
- Lines: **Human** = yes/no accuracy on human-authored GIFs; **Gemini-2.5-Pro** = yes/no on Gemini-made GIFs. Easy/hard pairwise protocols are in Figure 4.
- Input:
  [TASK] Human cue identification (yes/no)

  [CUE] `nod_yes` — Repeatedly nod to communicate yes or agreement.
  [EMBODIMENT] `humanoid`
  [VIDEO] Rendered GIF from a `human-authored` or `Gemini-made` Motion Code.

  [QUESTION] If shown this rendered GIF, is it recognizable as that cue? Answer yes or no.

| Embodiment | Human (authored) | Gemini-2.5-Pro | Trajectory sketch |
|---|---:|---:|---:|
| Humanoid | 100.0% | 53.3% | 70.0% |
| Tiago | 96.7% | 56.7% | 83.3% |
| Single Arm | 93.3% | 56.7% | 73.3% |
| Stretch | 100.0% | 56.7% | 93.3% |
| Roomba | 96.7% | 40.0% | 73.3% |
| **Overall** | **97.3%** | **52.7%** | **78.7%** |

---

## Generation

### Not perfectly recognizable alone, but distinguishable from foils

**Figure 4.** Human cue identification · yes/no vs easy-neg / hard-neg
- Graph: `data/results/evaluation/embodiment_cue_id/cueid_human_binary_acc_by_platform_gemini_made.png`
- Finding: On the same Gemini-made GIFs, humans separate a cue from a far foil (easy-neg) far better than they accept the video as that cue in yes/no—so generation lands in a near-miss band, not random motion.
- **Human**: yes/no on human-authored GIFs (same question as Figure 1).
- **Gemini-2.5-Pro**: yes/no on Gemini-made GIFs.
- **Trajectory sketch**: five-shot vector trajectory generation with hand + elbow Cartesian priority and intermediate wrist easing.
- **Easy-neg**: forced choice between the target cue and a far foil (lowest attribute F1 to human-authored). **Hard-neg**: target vs a near foil (highest non-exact F1). AB/BA order balanced.
- Question for pairwise: which of the two cue labels matches the GIF? Chance = 50%.
- Input:
  [TASK] Human cue identification — easy-neg / hard-neg pairwise

  [VIDEO] Gemini-made GIF for cue `nod_yes` on `humanoid`.
  [OPTION A] `nod_yes`
  [OPTION B] `shake_no` (easy-neg foil) or a near foil (hard-neg)

  [QUESTION] Which of the two cue labels matches the GIF? Answer A or B.

| Embodiment | Human (authored) | Gemini yes/no | Trajectory yes/no | Easy-neg | Hard-neg |
|---|---:|---:|---:|---:|---:|
| Humanoid | 100.0% | 53.3% | 70.0% | 96.7% | 73.3% |
| Tiago | 96.7% | 56.7% | 83.3% | 86.7% | 73.3% |
| Single Arm | 93.3% | 56.7% | 73.3% | 76.7% | 60.0% |
| Stretch | 100.0% | 56.7% | 93.3% | 83.3% | 60.0% |
| Roomba | 96.7% | 40.0% | 73.3% | 80.0% | 63.3% |
| **Overall** | **97.3%** | **52.7%** | **78.7%** | **84.7%** | **66.0%** |

**Figure 5.** Generation (vs Baselines)
- Graph: `data/results/evaluation/embodiment_cue_id/cueid_human_cue_identification_with_traj_baselines.png`
- Finding: Trajectory sketching reaches 78.7% overall, outperforming Motion Code (52.7%), joint-angle (52.7%), and xyzrpy (48.0%) on every embodiment. Its strongest results are Stretch (93.3%) and Tiago (83.3%).
- All three generated methods use Gemini-2.5-Pro with five shots. Blank baseline cells were scored as `oo`; first character = joint, second = xyzrpy. `c7/humanoid` xyzrpy and `c25/tiago` xyzrpy are missing and excluded rather than imputed.

| Embodiment | Motion Code (Ours) | Joint angles | xyzrpy | Trajectory sketch |
|---|---:|---:|---:|---:|
| Humanoid | 53.3% (16/30) | 46.7% (14/30) | 20.7% (6/29) | **70.0% (21/30)** |
| Tiago | 56.7% (17/30) | 46.7% (14/30) | 37.9% (11/29) | **83.3% (25/30)** |
| Single Arm | 56.7% (17/30) | 43.3% (13/30) | 53.3% (16/30) | **73.3% (22/30)** |
| Stretch | 56.7% (17/30) | 73.3% (22/30) | 70.0% (21/30) | **93.3% (28/30)** |
| Roomba | 40.0% (12/30) | 53.3% (16/30) | 56.7% (17/30) | **73.3% (22/30)** |
| **Overall** | 52.7% (79/150) | 52.7% (79/150) | 48.0% (71/148) | **78.7% (118/150)** |

**Table 1. Generation Difference**

| Channel | Morphology | n | accurate | diff | wrong | missing | extra |
|---|---|---:|---:|---:|---:|---:|---:|
| Static | Human-like | 60 | 55.6 | 44.4 | 28.6 | 6.0 | 9.8 |
| ^ | Non-humanoid (−roomba) | 60 | **77.8** | 22.2 | 13.6 | 2.3 | 6.3 |
| Dynamic | Human-like | 60 | 40.7 | 59.3 | 22.8 | 14.9 | 21.6 |
| ^ | Non-humanoid | 90 | **51.5** | 48.5 | 24.3 | 11.7 | 12.5 |

- Finding: Human-like drifts from human-authored motion mainly on **Static** pose values (accurate 55.6, diff 44.4), while Non-humanoid is relatively solid on pose (77.8) but still loses more on **Dynamic** bag-match (accurate 51.5, diff 48.5)—and Human-like dynamic is worse still (40.7) because Gemini adds extra dyn attrs.
- **Human-like** = humanoid + tiago. **Non-humanoid** = single_arm + stretch (+ roomba for Dynamic only).
- **Static**: paired `pose↔pose` only; per cell, union of pose keys → accurate / wrong / missing / extra, normalize to 100, average cells. Roomba excluded (n≈0 paired pose).
- **Dynamic**: bag-of-attrs over move/path leaves (order/component ignored); same four-way rates. Roomba included.
- **diff** = wrong + missing + extra (= 100 − accurate).
- **Outer `hold(...)` / zero-distance hold-as-path**: folded into the preceding component’s `hold_time` (added) before scoring.
- **Speed / hold_time**: unspecified = **0**; if `|Δ| < 0.5`, count as accurate (not wrong).
- Example: `pose(dir=front, gripper=horiz)` → `pose(dir=up, gripper=horiz)` → accurate 50, wrong 50, missing 0, extra 0.
- Example: `pose(dir=front, gripper=horiz, z=30)` → `pose(dir=up, gripper=horiz, y=60)` → accurate 25, wrong 25, missing 25, extra 25 (dir wrong, z missing, y extra).
- Mean of those two cells → accurate 37.5, wrong 37.5, missing 12.5, extra 12.5.
- DetailsToggle: By platform

| Channel | Platform | n | accurate | diff | wrong | missing | extra |
|---|---|---:|---:|---:|---:|---:|---:|
| Static | humanoid | 30 | 54.8 | 45.2 | 28.4 | 6.4 | 10.4 |
| ^ | tiago | 30 | 56.3 | 43.7 | 28.8 | 5.5 | 9.3 |
| ^ | single_arm | 30 | 64.3 | 35.7 | 20.3 | 2.9 | 12.5 |
| ^ | stretch | 30 | 91.3 | 8.7 | 7.0 | 1.7 | 0.0 |
| Dynamic | humanoid | 30 | 46.6 | 53.4 | 19.8 | 17.9 | 15.8 |
| ^ | tiago | 30 | 34.9 | 65.1 | 25.8 | 12.0 | 27.3 |
| ^ | single_arm | 30 | 45.1 | 54.9 | 27.0 | 12.9 | 15.1 |
| ^ | stretch | 30 | 67.7 | 32.3 | 13.1 | 6.7 | 12.5 |
| ^ | roomba | 30 | 41.7 | 58.3 | 33.0 | 15.5 | 9.8 |

**Table 2. Frequent Attribute Difference**

| Morphology | Channel | n | #1 | #2 | #3 | #4 | #5 |
|---|---|---:|---|---|---|---|---|
| Human-like | Static | 219 | Height (z) (20%) | Elbow (16%) | Reach (x) (15%) | Lateral (y) (14%) | Direction (12%) |
| ^ | Dynamic | 308 | Speed (16%) | Axis (14%) | Degrees (11%) | Distance (10%) | Hold time (9%) |
| Non-humanoid | Static | 88 | Height (z) (28%) | Lateral (y) (23%) | Reach (x) (20%) | Direction (12%) | Gripper (9%) |
| ^ | Dynamic | 444 | Speed (19%) | Path shape (11%) | Hold time (10%) | Radius (10%) | Degrees (9%) |

- Finding: Static diffs are mostly EE pose coordinates (z / y / x). Dynamic diffs are led by **Speed**; after folding outer holds into the previous component and treating unspecified hold as 0 with a **0.5s** cut, Hold time is no longer the top Dynamic error (HL 9%, NH 10%)—the remaining hold errors are mostly on/off swaps (e.g. 1.5↔0).
- Rows = morphology × channel; `%` = share among **diff** atoms only under Table 1 scoring.
- Hold instance check (HL, after fold): exact 59 / small `|Δ|<0.5` 8 / meaningful `|Δ|≥0.5` 26 (max `|Δ|=2.0`; top pattern `1.5→0`).

**Table 3. Pose space**

| Embodiment | total | pose | move | path | hold |
|---|---:|---:|---:|---:|---:|
| Humanoid | 25 | 10 | 6 | 8 | 1 |
| Tiago | 26 | 9 | 6 | 10 | 1 |
| Stretch | 24 | 9 | 4 | 10 | 1 |
| Single Arm | 19 | 6 | 4 | 8 | 1 |
| Roomba | 10 | 1 | 0 | 8 | 1 |

- Finding: Human-like body allows larger action space—more ways to get it wrong: humanoid/Tiago expose ~25–26 schema leaves vs Roomba’s 10, with much of the gap in pose.
- Source: `data/seed/yml/motion_code_embodiment.yaml`

**Script length**

**Figure 6.** Static / Dynamic Components Share
- Graph: `data/results/evaluation/plots/gt_static_dynamic_component_share.png`
- Finding: After ignoring bare `pose(dir=front)`, Roomba is **100% dynamic** for both human-authored and Gemini with the same mean dyn length (**2.2**)—humans do not write longer dynamic scripts; Gemini’s mismatch is channel choice / detail, not “more components.”
- Left = human-authored (GT); right = Gemini-2.5-Pro. Same 30 cues × embodiment.
- Per cue, share of **static** (`pose`) vs **dynamic** (`move`+`path`); average over cues. Bare `pose(dir=front)` ignored. `hold` type excluded; hold-as-path still counts as dynamic.
- Bar labels: `share% (mean component count)`.

---

## Knowledge across embodiments

### Models fill blanked human-authored attributes.

**Figure 2.** Blank accuracy · static vs dynamic
- Graph: `data/results/evaluation/plots/h4_blank_stretch_32b_static_dynamic.png`
- Finding: Given a nearly complete human-authored motion config, Qwen2.5-VL-32B recovers a blanked attribute well across embodiments; pose (static) and move/path (dynamic) are similarly easy (~75% vs ~72%).
- Task: blank-config multiple choice. One leaf attribute in a human-authored motion config is replaced by `<BLANK>`; the model chooses A or B.
- Prompt includes the cue meaning, the per-embodiment Motion Code schema (English glossary), and the full config with that one blank.
- **Static** = pose fields (`direction`, gripper, `left_arm`, `pose.*`). **Dynamic** = move/path fields (`repetition`, `path.*`). `path.x_sign` excluded. Roomba has essentially no static pose items.
- Bars pool original + answer-order counterbalanced packs. Black = all scored blanks; hatched = static; white = dynamic.
- Input:
  [SYSTEM] You are an expert in embodied robot gesture motion codes. Recover the missing value from cue semantics, the embodiment schema, and the full motion config context.

  [GESTURE MEANING]
  `Repeatedly nod to communicate yes or agreement.`

  [EMBODIMENT SCHEMA — `humanoid`]
  (English glossary of pose / move / path / hold fields for this robot)

  [INSTRUCTION]
  The full human-authored motion config for gesture `nod_yes` on the `humanoid` embodiment is shown below. One leaf attribute is replaced with `<BLANK>`. Recover the value of `gripper_orientation` for the 1st pose component.

  [FULL CONFIG]
  [
    {
      "type": "pose",
      "duration": 0.6,
      "parameters": {
        "pose": {
          "head": "center",
          "torso_height": "mid",
          "arm_position": "still",
          "gripper_orientation": "<BLANK>"
        }
      }
    },
    {
      "type": "movement",
      "parameters": {
        "movement": {
          "joints": [{"joint": "neck", "axis": "y", "degrees": [-15, 15]}],
          "repetition": 3,
          "speed": "mid",
          "hold_time": 0.2
        }
      }
    }
  ]

  [CHOICES]
  A. `vertical`
  B. `horizontal`

| Embodiment | All | Static | Dynamic |
|---|---:|---:|---:|
| Humanoid | 74.3 | 73.0 | 77.4 |
| Tiago | 76.3 | 76.1 | 76.9 |
| Single Arm | 77.3 | 77.6 | 76.3 |
| Stretch | 70.6 | 72.0 | 66.2 |
| Roomba | 67.0 | — | 65.6 |
| **Overall** | **74.1** | **74.8** | **72.3** |

### Blanking more context does not collapse accuracy

**Figure 3.** Blank accuracy vs hardness (k / component)
- Graph: `data/results/evaluation/plots/h4_hardness_32b_embodiment_heatmap.png`
- Finding: Accuracy stays flat as we blank more neighboring fields (k=1→5) or the whole component shell—so the model is not only exploiting a single nearby token.
- Same blank-config MCQ as Figure 2, but **k** controls how many additional fields are blanked in the same config.
- **k=1**: only the scored attribute is blank. **k=2…5**: that attribute plus k−1 other blanks. **component**: every leaf inside the pose/move/path component is blanked.
- Cell values = 32B accuracy (%). Same stretch-set platforms and orig+CB pooling as Figure 2.
- Input:
  [SYSTEM] You are an expert in embodied robot gesture motion codes. Recover the missing value from cue semantics, the embodiment schema, and the full motion config context.

  [GESTURE MEANING]
  `Repeatedly nod to communicate yes or agreement.`

  [EMBODIMENT SCHEMA — `humanoid`]
  (English glossary of pose / move / path / hold fields for this robot)

  [INSTRUCTION]
  The full human-authored motion config for gesture `nod_yes` on the `humanoid` embodiment is shown below. Recover the value of `gripper_orientation` when neighboring fields are also blanked (`k=3` or `component`).

  [FULL CONFIG · component blank]
  [
    {
      "type": "pose",
      "duration": 0.6,
      "parameters": {
        "pose": {
          "head": "<BLANK>",
          "torso_height": "<BLANK>",
          "arm_position": "<BLANK>",
          "gripper_orientation": "<BLANK>"
        }
      }
    },
    {
      "type": "movement",
      "parameters": {
        "movement": {
          "joints": [{"joint": "<BLANK>", "axis": "<BLANK>", "degrees": "<BLANK>"}],
          "repetition": "<BLANK>",
          "speed": "<BLANK>",
          "hold_time": "<BLANK>"
        }
      }
    }
  ]

  [CHOICES]
  A. `vertical`
  B. `horizontal`

| Tag | Humanoid | Tiago | Single Arm | Stretch | Roomba |
|---|---:|---:|---:|---:|---:|
| k=1 | 74.3 | 76.3 | 77.3 | 70.6 | 67.0 |
| k=2 | 74.7 | 76.9 | 77.6 | 73.9 | 64.0 |
| k=3 | 74.0 | 75.6 | 75.0 | 71.3 | 64.0 |
| k=4 | 74.3 | 76.9 | 76.0 | 68.7 | 62.0 |
| k=5 | 74.0 | 77.5 | 76.3 | 72.3 | 68.0 |
| component | 70.8 | 75.9 | 77.6 | 66.5 | 62.0 |

### Blank Accuracy · morphology × channel

**Figure 7.** Blank Accuracy
- Graph: `data/results/evaluation/plots/h4_blank_acc_morphology_static_dynamic_core_sub_32b.png`
- Finding: Core blank accuracy stays mid-70s on Human-like static/dynamic and Non-humanoid static; the drop is **Non-humanoid Dynamic** (core 70.0). Non-humanoid **sub** remains higher on static (87.5) than dynamic (65.2); Human-like sub omitted (—).
- Four groups: (Human-like / Non-humanoid) × (static / dynamic). Bars = **core** (all groups) + **sub** (Non-humanoid only).
- Each human-authored component is labeled **core** (needed for cue identity) or **submotion** (flourish / prep).
- Scores: `data/results/evaluation/plots/h4_blank_acc_morphology_static_dynamic_core_sub_32b.json`

**Table · Blank Accuracy**
- Layout: transpose + multicol by morphology · rows = core / sub · col groups = Human-like | Non-humanoid
- RawHtml:
  <table>
  <thead>
  <tr>
    <th class="col-text" rowspan="2"></th>
    <th class="col-num" colspan="2">Human-like</th>
    <th class="col-num" colspan="2">Non-humanoid</th>
  </tr>
  <tr>
    <th class="col-num">Static</th>
    <th class="col-num">Dynamic</th>
    <th class="col-num">Static</th>
    <th class="col-num">Dynamic</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td class="col-text">core</td>
    <td class="col-num">74.7 (442)</td>
    <td class="col-num">76.6 (158)</td>
    <td class="col-num">72.4 (388)</td>
    <td class="col-num">70.0 (200)</td>
  </tr>
  <tr>
    <td class="col-text">sub</td>
    <td class="col-num">—</td>
    <td class="col-num">—</td>
    <td class="col-num">87.5 (80)</td>
    <td class="col-num">65.2 (46)</td>
  </tr>
  </tbody>
  </table>

---

## Identification across embodiments

### VLMs lean on static poses, so they struggle with embodiment-extended gestures

**Figure 8.** VLM cue-ID · pairwise vs yes/no (Gemini judge)
- Graph: `data/results/evaluation/embodiment_cue_id/cueid_pairwise_acc_by_platform.png`
- Finding: VLMs lean on **static poses** for cue identity, so they struggle when the gesture is **embodiment-extended** (non-humanoid dynamics / Roomba near chance). As a pairwise judge Gemini-2.5-Pro still tracks morphology—strong on humanoid, weak on Roomba—but as a yes/no verifier on human-authored videos it collapses (overall 16%), blocking iterative “judge then regenerate.”
- Judge model: Gemini-2.5-Pro (thinking budget 128).
- **Pairwise**: easy-neg (target vs far foil), AB+BA; videos = human-authored or Gemini-made.
- **Yes/No**: match-only (is this GIF the cue?); videos = human-authored (N=150). Chance = 50%.
- Overall: pairwise human-authored 69.0% (207/300); pairwise Gemini-made 65.3% (196/300); yes/no human-authored **16.0%** (24/150).
- Input:
  [TASK] VLM cue identification — pairwise (easy-neg) or yes/no

  [VIDEO] Rendered GIF from `human-authored` (pairwise or yes/no) or `Gemini-made` (pairwise) on `humanoid`.
  [PAIRWISE] A=`nod_yes` vs B=`shake_no` (far foil) → answer A or B.
  [YES/NO] Cue=`nod_yes` → answer yes or no (match-only).

| Protocol · source | Overall | Humanoid | Tiago | Stretch | Single Arm | Roomba |
|---|---:|---:|---:|---:|---:|---:|
| Pairwise · Human-authored | 69.0 | 96.7 | 80.0 | 61.7 | 63.3 | 43.3 |
| Pairwise · Gemini-made | 65.3 | 90.0 | 68.3 | 60.0 | 65.0 | 43.3 |
| Yes/No · Human-authored | 16.0 | 20.0 | 43.3 | 3.3 | 6.7 | 6.7 |

**Table 4. Regen vs human-authored · Generation Difference (Human-NO)**
- Finding: Regenerating Gemini-made Motion Codes that humans judged NO does **not** reliably move them closer to human-authored codes—Human-like Dynamic gains a little (e.g. humanoid **+9.4**), but Non-humanoid often worsens (stretch Static −18.5, Dynamic −12.7).
- Set: Human yes/no **NO** on Gemini-made videos (71/150). Compare original Gemini vs regen Motion Codes to GT with Table 1 Generation Difference (fold outer hold; speed/hold tol 0.5).
- Cell = regen accurate (Δ vs original Gemini); **bold** = accuracy increased after regen. Roomba has Dynamic only.
- Input:
  [TASK] Motion Code regeneration (human-NO Gemini-made)

  [VIDEO] Gemini-made MP4 that a human judged NOT identifiable as the cue.
  [CURRENT CODE] Gemini Motion Code for that cell.
  [CUE] e.g. `fist_pump` + description.
  [SCHEMA] embodiment YAML for `stretch` / `tiago` / …

  [OUTPUT] Revised Motion Code (pose(…)+…); optional IdentifiableQ / ChangeSummary.

| Platform | n(NO) | Static | Dynamic |
|---|---:|---|---|
| humanoid | 13 | 40.5 (−2.9) | **35.6 (+9.4)** |
| tiago | 14 | **49.7 (+4.5)** | **32.2 (+2.1)** |
| single_arm | 13 | 55.9 (−1.2) | **32.8 (+0.9)** |
| stretch | 13 | 70.3 (−18.5) | 37.4 (−12.7) |
| roomba | 18 | — | 30.2 (−1.0) |

---

## Protocol notes

- H4 blank-config: per-embodiment schema YAML glossary (English), original + counterbalanced pooled; Qwen2.5-VL-32B; platforms humanoid, tiago, single_arm, stretch, roomba.
- `path.x_sign` excluded from H4 overall totals (roomba-only).
- Rebuild chain: `build_h4_binary_mcq.py` → blank → glossary → hardness → `label_gt_core_submotion.py`.
