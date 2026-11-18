# Balanced human-evaluation decks

- `L01`–`L10`: LLM-authored motions, 25 questions per deck (5 cues × 5 morphologies).
- `H01`–`H05`: human-authored motions, 30 questions per deck (6 cues × 5 morphologies).
- Every consecutive five-question block contains Unitree G1, TIAGo, Single Arm (IIWA), Stretch, and Roomba exactly once.
- Every assigned cue appears once on every morphology within its deck, enabling within-cue morphology comparisons even when only some decks are returned.
- A–D are randomized hard-negative choices with balanced correct-answer positions. E is always `Skip / 건너뛰기` and is never the keyed answer.
- English and Korean meanings are shown for every option. The answer is not displayed in any PPTX.
- Randomization seed: `2026091502`.

Keep the answer-key JSON files separate from evaluators.

