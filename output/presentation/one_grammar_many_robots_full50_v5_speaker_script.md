# One Grammar, Many Robots — 174-Second Presentation Script

## 0:00–0:06 · Slide 1 — Opening

What makes a robot gesture recognizable, even when the robot does not look like us? PaddleClaw says hi—and introduces the question behind this work.

## 0:06–0:13 · Slide 2 — Motivation

The same communicative meaning must survive very different bodies. A humanoid can beckon with a hand, while a Roomba must recreate that evidence through translation and rotation.

## 0:13–0:22 · Slide 3 — Three attributes

We describe that evidence using three recognition-critical attributes: orientation—where a carrier points; location—where it sits relative to the body; and movement—how it changes over time.

## 0:22–0:31 · Slide 4 — Retargeting and rematerialization

This gives us two transfer strategies. Retargeting preserves the coarse motion and maps it to another carrier. Rematerialization rebuilds the cue using whatever motion channels the new body provides.

## 0:31–0:41 · Slide 5 — Method

Our method is a symbolic Motion Program. A morphology schema defines available parts and ranges. A human or language model writes typed pose, movement, and path primitives, and a deterministic compiler renders robot-specific trajectories.

## 0:41–0:43 · Slide 6 — Evaluation scope

We evaluate fifty cues on five robots, with both LLM- and human-authored programs, and extend the same grammar to three synthetic morphologies.

## 0:43–1:58 · Slides 7–56 — Full 50-cue gallery

Here is the complete fifty-cue catalog. Each page shows the same LLM-authored cue across Unitree G1, TIAGo, IIWA, Stretch, and Roomba. I will not read every label; instead, notice what remains stable and what must change. Humanoid bodies often preserve familiar hand and head geometry. A single arm must translate head or bilateral actions into end-effector motion. Stretch combines arm extension with base movement, while Roomba relies almost entirely on planar translation and yaw.

Across nodding, pointing, beckoning, greeting, depicting shapes, and affective actions, recognizable examples preserve the cue's critical orientation, body-relative location, and temporal structure. Difficult examples reveal the limits of literal imitation. When a robot lacks the original carrier, success depends on selecting a new carrier and rebuilding equivalent perceptual evidence. These examples are therefore not only outputs; they expose how the grammar separates meaning from embodiment.

## 1:58–2:03 · Slide 57 — Human versus LLM

Human authors preserve more recognition-critical detail in difficult cases, especially hand orientation and morphology-aware rematerialization.

## 2:03–2:13 · Slide 58 — Human recognition

The human-recognition study shows the same pattern across all five embodiments: human-authored Motion Programs are consistently more recognizable than LLM-authored programs, although performance remains well above chance in many settings.

## 2:13–2:28 · Slides 59–60 — Synthetic robots

The grammar also transfers to unfamiliar bodies: PaddleClaw, SpineCrab, and TrilegCrown. For the same beckoning cue, each robot recruits a different carrier and motion structure. Thirty of fifty cues require this kind of morphology-conditioned rematerialization.

## 2:28–2:40 · Slides 61–63 — Failure cases

The failures are systematic. First, the semantic target may be correct while hand geometry is wrong. Second, the chosen carrier may move from an ambiguous location. Third, constrained robots can collapse distinct meanings into the same repeated base trajectory. These are representational failures, not merely rendering noise.

## 2:40–2:46 · Slide 64 — Takeaway

The main takeaway is that recognizable gesture is a problem of evidence. Motion Programs make that evidence explicit enough to preserve it—or rebuild it—across radically different bodies.

## 2:46–2:54 · Slide 65 — Further directions

Next, we want controllable affect and traits, perceptual feedback that repairs errors before deployment, and automatic discovery of useful carriers on unseen morphologies. Thank you.
