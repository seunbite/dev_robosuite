# One Grammar, Many Robots — ICRA Video Narration
**Deck:** `one_grammar_many_robots_full50_v5_bright_paddleclaw.pptx` (32 slides)
**Target runtime:** ~173 s (limit 180 s)
**Rate assumption:** ~155 wpm / 2.6 words per second. Recheck after the first TTS pass.

---

## Part 1 · Framing (Slides 1–7) — 52 s

**Slide 1 — What makes a robot gesture recognizable?** *(7 s)*
> Gesture generation turns communicative intent into robot motion. This work asks how that meaning survives very different bodies.

**Slide 2 — Motivation** *(7 s)*
> A humanoid can beckon with a hand. A Roomba must rebuild the same evidence using translation and yaw.

**Slide 3 — Recognition-critical attributes (humanoid)** *(7 s)*
> We describe that evidence with three attributes: orientation, body-relative location, and movement.

**Slide 4 — Recognition-critical attributes (IIWA)** *(6 s)*
> The same decomposition holds for a single arm — gripper direction, height, and repetition each carry evidence.

**Slide 5 — Retargeting and rematerialization** *(8 s)*
> Transfer takes two forms. Retargeting maps the coarse motion onto another carrier. Rematerialization rebuilds the cue from the channels the body offers.

**Slide 6 — Symbolic Motion Programs** *(11 s)*
> A morphology schema declares the available carriers. A human or L-L-M author writes typed primitives, and a deterministic compiler renders the robot-specific trajectory.

**Slide 7 — Evaluation scope** *(7 s)*
> Fifty cues, five robots, four hundred Motion Programs, and one hundred sixty-five recognition trials.

---

## Part 2 · Fifty-cue gallery (Slides 8–25) — 72 s
*Each slide ≈ 4 s. Keep the GIFs visible for the full slide duration; the line is short on purpose.*

**8 — Nod Yes** *(4 s)*
> Repeated vertical rhythm survives on every body, with or without a head.

**9 — Shake Head No** *(4 s)*
> Lateral alternation stays readable when the head is replaced by a base.

**10 — Cover Yawn Sleepy** *(4 s)*
> Here the cue depends on bringing a carrier toward the mouth region.

**11 — Rub Eye Tired** *(4 s)*
> Small repeated motion near the eye supplies the critical evidence.

**12 — Point You** *(4 s)*
> Forward direction and extension preserve the deictic target.

**13 — Point Self** *(4 s)*
> Reversing that direction returns the reference to the robot.

**14 — Beckon Come Here** *(4 s)*
> Repeated inward motion keeps the invitation to approach.

**15 — Shrug, I don't know** *(4 s)*
> Raised, outward motion conveys uncertainty across all five action spaces.

**16 — Arms Open Welcome** *(4 s)*
> Expansion away from the body retains the welcoming intent.

**17 — Hands Up Surrender** *(4 s)*
> High, open carriers hold the surrender cue.

**18 — Stretch Arms Up** *(4 s)*
> Upward extension separates stretching from a static high pose.

**19 — Scratch Head Confused** *(4 s)*
> Repeated motion near the head reads as thought and uncertainty.

**20 — Visor Search** *(4 s)*
> A raised carrier sweeps across the viewing direction.

**21 — Slow Down Request** *(4 s)*
> A downward-facing carrier and a controlled descent signal reduced speed.

**22 — Depict Tree** *(4 s)*
> Depictive cues shift the work to shape: vertical structure and branching.

**23 — Depict Lion** *(4 s)*
> A forceful forward reach stands in for the animal.

**24 — Depict Flying Bird** *(4 s)*
> Repeated vertical flapping remains the defining motion everywhere.

**25 — Frustration Down Throw** *(4 s)*
> And a sharp downward release carries affect rather than reference.

---

## Part 3 · Results and discussion (Slides 26–32) — 49 s

**Slide 26 — Human and L-L-M authoring** *(7 s)*
> Human authors preserve more recognition-critical detail, especially hand orientation and morphology-aware rematerialization.

**Slide 27 — Human recognition** *(7 s)*
> In the recognition study, human-authored Motion Programs remain more recognizable across all five embodiments.

**Slide 28 — Three synthetic morphologies** *(7 s)*
> The grammar extends to unfamiliar bodies: PaddleClaw, SpineCrab, and TrilegCrown recruit unusual parts as carriers.

**Slide 29 — Morphology-conditioned rematerialization** *(7 s)*
> The same beckon becomes a different motion structure on each body. Thirty of fifty cues need this.

**Slide 30 — Failure cases: attribute error** *(7 s)*
> Failures are systematic: many keep the action but corrupt one attribute — orientation, location, or movement.

**Slide 31 — Failure cases: constrained-body collapse** *(7 s)*
> On Roomba, distinct meanings collapse into one fore-and-aft trajectory, with no rotation, arc, or lateral cue to separate them.

**Slide 32 — Further directions** *(7 s)*
> Recognizable gesture is a problem of evidence. Next: controllable affect, perceptual feedback, and automatic carrier discovery. Thank you.

---

## Pronunciation
- **IIWA** → "ee-vah"
- **TIAGo** → "tee-AH-go"
- **L-L-M** → spelled out
- **Unitree G1** → "YOO-nee-tree G-one"

## Timing notes
- **Rev 2 (TTS-measured).** The first cut ran 195 s once synthesized, so eight lines on slides 6, 7, 26, 28, 30, 31, 32 were shortened rather than speeding the voice up. Measured total is now ~172 s.
- If TTS runs long, trim the gallery first: slides 16, 17, 20, 23 can drop to a 6–7 word line or go silent.
- Explanatory slides: hold TTS duration + 0.4–0.7 s. Gallery slides: hold at least 4 s regardless of line length.
