# One Grammar, Many Robots — narration (28-slide deck, Rev 3)

Runtime: **160.9 s** (Ryan) / **161.7 s** (Lessac). Limit is 180 s, so there is ~19 s of headroom.

Rev 3 changes: the deck dropped Beckon, Slow Down, Flying Bird and Frustration Down Throw, so those four lines are gone. Slide 23's claim was corrected to match the figure (see note at the bottom).

**Slide 1 — What makes a robot gesture recognizable?**
> Gesture generation turns communicative intent into robot motion. This work asks how that meaning survives very different bodies.

**Slide 2 — Motivation**
> A humanoid can beckon with a hand. A Roomba must rebuild the same evidence using translation and yaw.

**Slide 3 — Recognition-critical attributes (humanoid)**
> We describe that evidence with three attributes: orientation, body-relative location, and movement.

**Slide 4 — Recognition-critical attributes (IIWA)**
> The same decomposition holds for a single arm — gripper direction, height, and repetition each carry evidence.

**Slide 5 — Retargeting and rematerialization**
> Transfer takes two forms. Retargeting maps the coarse motion onto another carrier. Rematerialization rebuilds the cue from the channels the body offers.

**Slide 6 — Symbolic Motion Programs**
> A morphology schema declares the available carriers. A human or L-L-M author writes typed primitives, and a deterministic compiler renders the robot-specific trajectory.

**Slide 7 — Evaluation scope**
> Fifty cues, five robots, four hundred Motion Programs, and one hundred sixty-five recognition trials.

**Slide 8 — 01 Nod Yes**
> Repeated vertical rhythm survives on every body, with or without a head.

**Slide 9 — 02 Shake Head No**
> Lateral alternation stays readable when the head is replaced by a base.

**Slide 10 — 07 Cover Yawn Sleepy**
> Here the cue depends on bringing a carrier toward the mouth region.

**Slide 11 — 08 Rub Eye Tired**
> Small repeated motion near the eye supplies the critical evidence.

**Slide 12 — 09 Point You**
> Forward direction and extension preserve the deictic target.

**Slide 13 — 10 Point Self**
> Reversing that direction returns the reference to the robot.

**Slide 14 — 16 Shrug Idk**
> Raised, outward motion conveys uncertainty across all five action spaces.

**Slide 15 — 19 Arms Open Welcome**
> Expansion away from the body retains the welcoming intent.

**Slide 16 — 20 Hands Up Surrender**
> High, open carriers hold the surrender cue.

**Slide 17 — 21 Stretch Arms Up**
> Upward extension separates stretching from a static high pose.

**Slide 18 — 31 Scratch Head Confused**
> Repeated motion near the head reads as thought and uncertainty.

**Slide 19 — 43 Visor Search**
> A raised carrier sweeps across the viewing direction.

**Slide 20 — 26 Depict Tree**
> Depictive cues shift the work to shape: vertical structure and branching.

**Slide 21 — 28 Depict Lion**
> A forceful forward reach stands in for the animal.

**Slide 22 — Human and LLM authoring**
> Human authors preserve more recognition-critical detail, especially hand orientation and morphology-aware rematerialization.

**Slide 23 — Human recognition**
> In the recognition study, human-authored Motion Programs lead on four of the five robots, with the largest gap on Roomba.

**Slide 24 — Three synthetic morphologies**
> The grammar extends to unfamiliar bodies: PaddleClaw, SpineCrab, and TrilegCrown recruit unusual parts as carriers.

**Slide 25 — Morphology-conditioned rematerialization**
> The same beckon becomes a different motion structure on each body. Thirty of fifty cues need this.

**Slide 26 — Failure cases: attribute error**
> Failures are systematic: many keep the action but corrupt one attribute — orientation, location, or movement.

**Slide 27 — Failure cases: constrained-body collapse**
> On Roomba, distinct meanings collapse into one fore-and-aft trajectory, with no rotation, arc, or lateral cue to separate them.

**Slide 28 — Further directions**
> Recognizable gesture is a problem of evidence. Next: controllable affect, perceptual feedback, and automatic carrier discovery. Thank you.

---

## Note on slide 23

The figure `figure_human_eval_4afc_partial_latest.png` shows Stretch at 44.4 % human-authored versus 46.7 % LLM-authored — the LLM wins that one. The deck's subtitle still reads "remain more recognizable across all five robot embodiments", which the figure contradicts.

The narration now says **four of the five robots, with the largest gap on Roomba**. The slide subtitle still needs the same fix in PowerPoint.
