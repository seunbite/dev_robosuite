# Paper next (status)

## Done in `root.tex`
1. Appendix Table (`tab:cue-categories-full`) is now the **30-cue screening subset**
   (n=4/5/10/11 by required body part), not the 90-cue inventory. Table~I caption and
   the §5.A reference were repointed to the supplementary material accordingly.
2. §4 now specifies the grammar: parameter domains for `pose`/`movement`/`path`,
   connector semantics (`A & B` == `order=together` on A, resolved pairwise),
   the 206-pose G1 feasible bank and its four selection tiers, and the fact that
   one converter is hand-written per morphology.
3. §4.C states the human-vs-LLM revision-budget asymmetry explicitly.
4. §5.A documents the 4AFC hard-negative rule (1 nearest Motion-Program foil +
   2 meaning neighbours from a curated confusion map / same communicative family,
   fixed-seed shuffle) and adds a Human study protocol paragraph.
5. §6 adds `tab:remat-strategies`: all 150 LLM-authored programs coded into
   carrier / tracing / additive / none by `classify_rematerialization.py`.
6. §6 restores `tab:nonhumanoid-olm` (the former Table IV): the Sec.V attribute
   metrics run on IIWA / Stretch / Roomba, 3 generators + random.
   Fixed an over-count in `score_nonhumanoid_table4.score_palm`, which credited
   the GT gripper token if it appeared at ANY pose rather than the scored one
   (IIWA Plm 75.0 -> 71.4; Stretch unaffected, its 100% is a real 26/26).

## Rematerialization strategy distribution (% of 30 cues)
| Body | Carrier | Tracing | Additive | None |
|---|---:|---:|---:|---:|
| Tiago | 47 | 7 | 20 | 27 |
| IIWA | 40 | 7 | 10 | 43 |
| Stretch | 30 | 13 | 20 | 37 |
| Roomba | 20 | 27 | 33 | 20 |
| *strategy-conditioned* | | | | |
| IIWA | 37 | 3 | 37 | 23 |
| Stretch | 57 | 3 | 37 | 3 |
| Roomba | 27 | 33 | 37 | 3 |

Strategy conditioning run on all three constrained bodies via
`adhoc/generation/robotarm/generate_strategy_fewshot.py --body stretch|single_arm`
(Roomba keeps its own script). 30/30 parse on both new conditions.
General effect: `none` collapses (43->23, 37->3, 20->3) and additive converges
to 37% on all three. Body-specific effect: Stretch converts the freed mass into
carrier transfer (30->57), Roomba cannot (20->27) and traces instead, IIWA keeps
a 23% residue -- one chain must carry orientation, location and stroke at once.

Carrier transfer falls monotonically with channel loss (47→40→30→20). Strategy
conditioning does **not** restore carrier transfer; it removes the
non-rematerialized residue (none 20%→3%) and redistributes it to tracing and
additive. Validation: the coder reproduces all five strategy assignments the
paper names by hand (IIWA `shake_head_no`, Stretch `depict_big_bowl`,
Roomba `raise_fists_victory` / `nod_yes` / `shake_head_no`).

## Attribute agreement, constrained bodies (30 cues, vs human GT)
Computed but NOT in the paper (Table II removed on request); grid lives in
`motion_code/nonhumanoid30_table4_scores.md`, LaTeX in `nonhumanoid30_table4.tex`.
Gemini scores higher on IIWA/Stretch than on G1/Tiago (Rch 70.0/69.2 vs 47.2/49.1)
while IIWA carries the 43% `none` rate. Generator ordering is clean (Gemini >
32B > 7B on every column), unlike G1 where Qwen-32B led on reach. Restore as a
supplementary table.

## Experiments still to run (blocking)
1. **Human recognition, humanoid-like** — human- vs LLM-authored, G1 + Tiago.
   Recruited sample, per-clip replication, counterbalanced order, attention
   checks, CIs. Replaces the single-rater pilot (79/90 trials answered).
   Also yields the confusion matrix for (2).
2. **D_B vs human confusion** — correlate the typed active-channel distance with
   the confusion matrix from (1) (Mantel test). If it holds, the claim becomes
   "representation-space distance predicts human confusability", and Roomba /
   Spine Crab stop being anecdotes.
3. **Human recognition, constrained bodies** — IIWA, Stretch, Roomba, and
   Roomba-strategy-conditioned. Gives (a) the transfer effect, (b) recognition
   preserved per rematerialization strategy (joins `tab:remat-strategies`),
   (c) a measured aliasing rate.
4. **Baseline**: few-shot xyzrpy / joint through the context pipeline.
   `adhoc/generation/robotarm/run_context_baseline_traj.py`. Gemini 2.5 Pro
   returned 429 RESOURCE_EXHAUSTED (prepay credits depleted) for all 60 jobs.
   Re-run after topping up: `source APIKEY.sh && micromamba env robosuite python …`

## Open editorial
- Resolve the four `\TBD` red marks in `root.tex` (they list what each needs).
- **9 pages; the limit is 8 including references.** Appendix table → supplementary
  is the cheapest cut left.
