# One Grammar, Many Robots
## Narration draft for approval

Target runtime: approximately 178 seconds before final TTS measurement.

Timing rule after approval:

- Explanatory slides: actual TTS duration plus 0.4 to 0.7 seconds.
- Gesture gallery slides: short narration, then keep the GIFs visible until the slide reaches at least 4 seconds.
- No audio or video will be generated until the wording and pronunciation choices are approved.

## Opening sentence options

### Option A — direct and formal

Gesture generation is the task of turning communicative intent into motion that a robot can physically execute.

### Option B — meaning-centered

Gesture generation asks how a robot can express an intended meaning through the body and motion channels available to it.

### Option C — recognition-centered

In gesture generation, we translate communicative intent into robot motion while preserving the evidence people need to recognize.

## Page-by-page narration

### Slide 1 — What makes a robot gesture recognizable? — target 7 s

[Insert the selected opening sentence.] This work asks how that meaning can survive across very different robot bodies.

### Slide 2 — Motivation — target 7 s

A humanoid can beckon with a hand. A Roomba must rebuild the same communicative evidence through translation and rotation.

### Slide 3 — Recognition-critical attributes: humanoid — target 7 s

We describe that evidence through orientation, body-relative location, and movement. Different gestures depend on different combinations of these attributes.

### Slide 4 — Recognition-critical attributes: IIWA — target 7 s

The same decomposition applies to a single arm. Here, gripper direction, vertical position, and repeated motion contribute distinct evidence.

### Slide 5 — Retargeting and rematerialization — target 8 s

Retargeting maps a coarse motion onto another carrier. Rematerialization rebuilds the cue with the motion channels that the new body provides.

### Slide 6 — Symbolic Motion Programs — target 9 s

Our symbolic Motion Program separates intent from rendering. A morphology schema defines the available parts, an author specifies typed primitives, and a deterministic compiler produces the trajectory.

### Slide 7 — Evaluation scope — target 7 s

We study fifty cues on five evaluated robots, compare LLM and human authors, and extend the same grammar to three synthetic morphologies.

## Selected gesture gallery

Each gallery slide has a short line followed by time for the five GIFs to play. Minimum duration: 4 seconds per slide.

### Slide 8 — Nod yes

Repeated vertical rhythm preserves the meaning across all five bodies.

### Slide 9 — Shake head no

Lateral alternation remains readable even when the robot has no head.

### Slide 10 — Cover yawn sleepy

The cue depends on bringing a carrier toward the mouth region.

### Slide 11 — Rub eye tired

Small repeated motion near the eye provides the critical evidence.

### Slide 12 — Point you

Forward direction and extension preserve the deictic target.

### Slide 13 — Point self

Reversing the direction brings the reference back to the robot.

### Slide 14 — Beckon come here

Repeated inward motion preserves the invitation to approach.

### Slide 15 — Shrug, I do not know

Raised outward motion conveys uncertainty across the embodiments.

### Slide 16 — Arms open welcome

Expansion away from the body retains the welcoming intent.

### Slide 17 — Hands up surrender

High, open carriers preserve the surrender cue.

### Slide 18 — Stretch arms up

Upward extension distinguishes stretching from a static high pose.

### Slide 19 — Scratch head confused

Repeated motion near the head conveys uncertainty and thought.

### Slide 20 — Visor search

A raised carrier sweeps across the viewing direction.

### Slide 21 — Slow down request

A lowered palm and controlled descent signal reduced speed.

### Slide 22 — Depict tree

Vertical structure and branching motion depict a tree.

### Slide 23 — Depict lion

A forceful forward reach suggests the animal's presence.

### Slide 24 — Depict flying bird

Repeated vertical flapping remains the defining motion.

### Slide 25 — Frustration down throw

A sharp downward release conveys frustration.

## Results and discussion

### Slide 26 — Human and LLM authoring — target 8 s

Human authors preserve more recognition-critical detail in difficult cases, especially hand orientation and morphology-aware rematerialization.

### Slide 27 — Human recognition — target 8 s

[Insert the selected results sentence.]

### Slide 28 — Three synthetic morphologies — target 7 s

The grammar also transfers to unfamiliar bodies. PaddleClaw, SpineCrab, and TrilegCrown recruit unusual parts as communicative carriers.

### Slide 29 — Morphology-conditioned rematerialization — target 8 s

The same beckoning cue becomes a different motion structure on each synthetic body. Thirty of fifty cues require this type of rematerialization.

### Slide 30 — Failure cases: attribute error — target 8 s

Several failures preserve the general action while corrupting one critical attribute. Errors in orientation, location, or movement can make the intended cue ambiguous.

### Slide 31 — Failure cases: constrained-body collapse — target 8 s

On Roomba, distinct meanings often collapse into the same fore-and-aft trajectory. The model underuses rotation, arcs, and lateral motion that could separate them.

### Slide 32 — Further directions — target 7 s

Future work should add controllable affect, use perceptual feedback to repair errors, and discover effective carriers automatically on unseen morphologies. Thank you.

## Results sentence options for Slide 27

### Option A — direct comparison

Human-authored programs lead on four of five robots, with the largest recognition gap on Roomba.

### Option B — cautious interpretation

Human authors generally preserve more recognizable evidence, although the advantage is not uniform across embodiments.

### Option C — emphasize embodiment

The authorship gap changes with embodiment, and it becomes most pronounced when the robot offers fewer human-like motion channels.

## Pronunciation choices

### IIWA

- Option A: “ee-vah”
- Option B: spell out “I-I-W-A”

### TIAGo

- Recommended: “tee-AH-go”

### LLM

- Recommended: spell out “L-L-M”
