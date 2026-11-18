# One Grammar, Many Robots — narration Rev 5 (for approval)

Measured runtime **175.1 s** (Ryan voice, same timing rules as before). Limit 180 s.

## What changed from Rev 4

**Every line is now a complete sentence.** The gallery previously ran as clauses split across slides — a line ending in a comma, the next one picking it up in lowercase. That reads fine on paper but sounds unfinished when a slide cuts. Each slide now closes its own sentence while the grouping logic stays.

**Connective tissue was added at the joints, not everywhere.** The chain that now carries the argument:

- Slide 3 opens **"We therefore describe a cue by the evidence it carries"** — the attributes follow from the Roomba problem on slide 2.
- Slide 5 opens **"Transfer then takes two forms"**, following from the attributes.
- Slide 6 opens **"We therefore propose symbolic Motion Programs"** — the method is now explicitly motivated by the two strategies rather than just appearing.
- Slide 22 opens **"Those programs were all LLM-authored"**, which turns the gallery into the setup for the human comparison.
- Slide 24 opens **"It also extends to…"**, slide 26 opens **"Not every case succeeds"**, slide 28 opens **"Recognizable gesture is therefore a problem of evidence."**

Slides 9–21 inside the gallery were deliberately left without connectives, so the transitions stay audible instead of turning into filler.

**Sixteen lines were shortened** to pay for the above — the first draft of Rev 5 measured 201.7 s.

---

## Slide 6 (animated reveal — four stages, one per sentence)

1. We therefore propose symbolic Motion Programs. A schema declares each body's carriers.
2. An author, human or L L M, writes a program against it.
3. Programs use typed pose, movement, and path primitives.
4. A compiler then renders the trajectory deterministically.

---

**Slide 1 — What makes a robot gesture recognizable?**  · 6.5 s
> Gesture generation turns communicative intent into robot motion. How does that meaning survive very different bodies?

**Slide 2 — Motivation**  · 6.5 s
> A humanoid can beckon with a hand, but a Roomba must rebuild the same evidence using translation and yaw.

**Slide 3 — Recognition-critical attributes (humanoid)**  · 6.7 s
> We therefore describe a cue by the evidence it carries: orientation, body-relative location, and movement.

**Slide 4 — Recognition-critical attributes (IIWA)**  · 7.0 s
> The decomposition is not specific to humanoids. On a single arm, gripper direction, height, and repetition carry it.

**Slide 5 — Retargeting and rematerialization**  · 8.7 s
> Transfer then takes two forms. Retargeting maps the coarse motion onto another carrier; rematerialization rebuilds the cue from the new body's channels.

**Slide 6 — Symbolic Motion Programs**  · 16.3 s
> We therefore propose symbolic Motion Programs. A schema declares each body's carriers. An author, human or L L M, writes a program against it. Programs use typed pose, movement, and path primitives. A compiler then renders the trajectory deterministically.

**Slide 7 — Evaluation scope**  · 7.4 s
> We evaluate the grammar on fifty cues and five robots, with four hundred programs and one hundred sixty-five trials.

**Slide 8 — 01 Nod Yes**  · 6.4 s
> The examples that follow were authored by Gemini 2.5 Pro. A nod is simply vertical oscillation.

**Slide 9 — 02 Shake Head No**  · 5.7 s
> A head shake is the same structure turned horizontal, and both survive on bodies with no head.

**Slide 10 — 07 Cover Yawn Sleepy**  · 4.2 s
> Two face-directed cues come next. Yawning needs a carrier brought to the mouth.

**Slide 11 — 08 Rub Eye Tired**  · 4.0 s
> Fatigue needs small repeated motion near the eye instead.

**Slide 12 — 09 Point You**  · 4.4 s
> Deixis is the cleanest case, because direction alone carries the meaning.

**Slide 13 — 10 Point Self**  · 3.5 s
> Reverse that direction and the reference returns to the robot.

**Slide 14 — 16 Shrug Idk**  · 5.5 s
> Shrugging is harder: raised outward motion must read as uncertainty with no face to help.

**Slide 15 — 19 Arms Open Welcome**  · 6.0 s
> Three postural cues share one channel: expansion away from the body. Welcome opens outward.

**Slide 16 — 20 Hands Up Surrender**  · 3.5 s
> Surrender lifts the carriers high and open.

**Slide 17 — 21 Stretch Arms Up**  · 4.2 s
> Stretching adds the upward extension that separates it from a static pose.

**Slide 18 — 31 Scratch Head Confused**  · 3.5 s
> Near the head, repeated motion reads as thought.

**Slide 19 — 43 Visor Search**  · 3.7 s
> A raised carrier sweeping across the gaze reads as search.

**Slide 20 — 26 Depict Tree**  · 7.3 s
> Depiction is hardest, because the motion must draw the referent rather than point at it. A tree becomes vertical structure and branching.

**Slide 21 — 28 Depict Lion**  · 5.9 s
> A lion becomes a forceful forward reach. Here the grammar must rebuild evidence rather than copy it.

**Slide 22 — Human and LLM authoring**  · 7.5 s
> Those programs were all LLM-authored. Human authors preserve more detail in hand orientation and rematerialization.

**Slide 23 — Human recognition**  · 5.5 s
> Human-authored programs lead on four of the five robots, with the largest gap on Roomba.

**Slide 24 — Three synthetic morphologies**  · 7.5 s
> It also extends to unfamiliar bodies, where PaddleClaw, SpineCrab, and TrilegCrown recruit unusual carriers.

**Slide 25 — Morphology-conditioned rematerialization**  · 5.6 s
> The same beckon becomes a different motion structure on each, and thirty of fifty cues need this.

**Slide 26 — Failure cases: attribute error**  · 6.8 s
> Not every case succeeds, and the failures are systematic. Many preserve the action but corrupt one attribute.

**Slide 27 — Failure cases: constrained-body collapse**  · 7.4 s
> On Roomba, distinct meanings collapse into one fore-and-aft trajectory, with no rotation or arc to separate them.

**Slide 28 — Further directions**  · 7.8 s
> Recognizable gesture is therefore a problem of evidence. Next come controllable affect, perceptual feedback, and carrier discovery.

---

## Still to fix in the pptx

Slide 23's subtitle still reads "remain more recognizable across all five robot embodiments" while the figure shows Stretch at 44.4 % human versus 46.7 % LLM. The narration says "four of the five robots".