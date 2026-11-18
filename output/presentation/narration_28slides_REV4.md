# One Grammar, Many Robots — narration (28-slide deck, Rev 4)

Runtime **177.8 s** (Ryan) / **177.2 s** (Lessac). Limit 180 s.

## Rev 4 changes

- **Slide 6 now animates.** The four pipeline stages appear one at a time, each cued to its own narration phrase. The bottom example box appears with stage 4. This is done in the video only — the pptx is untouched.
- **Slide 4** no longer frames the decomposition as arm-specific; it now claims non-humanoid bodies generally, with the single arm as the instance.
- **The gallery (slides 8–21) was rewritten as connected prose.** Cues are grouped by shared mechanism instead of narrated one by one, and line lengths deliberately vary — a longer setup line carries a group, then short lines ride it out. Slide 8 opens with the Gemini 2.5 Pro attribution.
- Ten lines elsewhere were shortened to pay for the added time.

## Slide 6 phrase timing (Ryan)

| stage appears at | phrase |
|---|---|
| 0.0 s | 1. A morphology schema declares the carriers each body offers. |
| 3.6 s | 2. A human or L L M author writes a program against it, |
| 6.9 s | 3. using typed pose, movement, and path primitives, |
| 10.0 s | 4. which a deterministic compiler renders as a robot-specific trajectory. |

The reveal points are driven by the measured phrase durations, so re-synthesizing slide 6 re-derives them automatically.

---

**Slide 1 — What makes a robot gesture recognizable?**
> Gesture generation turns communicative intent into robot motion. How does that meaning survive very different bodies?

**Slide 2 — Motivation**
> A humanoid can beckon with a hand. A Roomba must rebuild the same evidence using translation and yaw.

**Slide 3 — Recognition-critical attributes (humanoid)**
> We describe that evidence with three attributes: orientation, body-relative location, and movement.

**Slide 4 — Recognition-critical attributes (IIWA)**
> The same decomposition holds for non-humanoid bodies: on a single arm, gripper direction, height, and repetition each carry evidence.

**Slide 5 — Retargeting and rematerialization**
> Transfer takes two forms. Retargeting maps the coarse motion onto another carrier; rematerialization rebuilds the cue from the channels the body offers.

**Slide 6 — Symbolic Motion Programs**
> A morphology schema declares the carriers each body offers. A human or L-L-M author writes a program against it, using typed pose, movement, and path primitives, which a deterministic compiler renders as a robot-specific trajectory.

**Slide 7 — Evaluation scope**
> Fifty cues, five robots, four hundred Motion Programs, and one hundred sixty-five recognition trials.

**Slide 8 — 01 Nod Yes**
> Everything from here was authored by Gemini 2.5 Pro and rendered on all five robots. A nod is simply vertical oscillation.

**Slide 9 — 02 Shake Head No**
> A head shake is the same structure turned horizontal, and both survive on bodies with no head.

**Slide 10 — 07 Cover Yawn Sleepy**
> Two face-directed cues next. Yawning needs a carrier brought to the mouth,

**Slide 11 — 08 Rub Eye Tired**
> and fatigue needs small repeated motion near the eye.

**Slide 12 — 09 Point You**
> Deixis is the cleanest case, where direction alone carries the meaning.

**Slide 13 — 10 Point Self**
> Reverse it, and the reference returns to the robot.

**Slide 14 — 16 Shrug Idk**
> Shrugging is harder: raised, outward motion must read as uncertainty with no face to help.

**Slide 15 — 19 Arms Open Welcome**
> Three postural cues share one channel, expansion away from the body. Welcome opens outward,

**Slide 16 — 20 Hands Up Surrender**
> surrender lifts high and open,

**Slide 17 — 21 Stretch Arms Up**
> and stretching adds upward extension, which is what separates it from a static pose.

**Slide 18 — 31 Scratch Head Confused**
> Near the head, repeated motion reads as thought,

**Slide 19 — 43 Visor Search**
> while a raised carrier sweeping across the gaze reads as search.

**Slide 20 — 26 Depict Tree**
> Depiction is hardest: the motion has to draw the referent, not point at it. A tree becomes vertical structure and branching,

**Slide 21 — 28 Depict Lion**
> a lion becomes a forceful forward reach. This is where the grammar must rebuild evidence rather than copy it.

**Slide 22 — Human and LLM authoring**
> Human authors preserve more recognition-critical detail, especially hand orientation and rematerialization.

**Slide 23 — Human recognition**
> In the recognition study, human-authored Motion Programs lead on four of the five robots, with the largest gap on Roomba.

**Slide 24 — Three synthetic morphologies**
> The grammar extends to unfamiliar bodies: PaddleClaw, SpineCrab, and TrilegCrown recruit unusual carriers.

**Slide 25 — Morphology-conditioned rematerialization**
> The same beckon becomes a different motion structure on each body. Thirty of fifty cues need this.

**Slide 26 — Failure cases: attribute error**
> Failures are systematic: many keep the action but corrupt one attribute, orientation, location, or movement.

**Slide 27 — Failure cases: constrained-body collapse**
> On Roomba, distinct meanings collapse into one fore-and-aft trajectory, with no rotation or arc to separate them.

**Slide 28 — Further directions**
> Recognizable gesture is a problem of evidence. Next: controllable affect, perceptual feedback, and automatic carrier discovery.

---

## Still to fix in the pptx

Slide 23's subtitle reads "remain more recognizable across all five robot embodiments", but the figure shows Stretch at 44.4 % human versus 46.7 % LLM. The narration says "four of the five robots"; the slide text does not match it yet.