# Literature map: language-executable communicative robot motion

This map contains 58 candidate references. It deliberately separates **core citations** (use in the two related-work paragraphs) from **supporting/context citations** (use only where a reviewer needs a specific comparison). Before submission, export and verify BibTeX/venue metadata for every work selected for the final bibliography.

## Recommended framing

**Problem.** We study the motion-abstraction bottleneck between an LLM's gesture knowledge and executable robot motion. A cue may demand a body part, body-relative location, end-effector direction, orientation, path, speed, repetition, timing, and bilateral coordination. The same cue can be realized differently on different bodies. The benchmark therefore evaluates preservation of **communicative effect**, not literal joint or pose equivalence.

**Deliberate scope.** The benchmark contains no objects, environment interaction, contact, force control, perception, or task planning. This excludes object-centric success as a confound and isolates language-to-motion representation.

## Two-paragraph Related Work draft

### Paragraph 1 — Gesture generation and communicative robot behavior

Robot gesture has long been specified with hand-authored behavior libraries and multimodal behavior languages, while data-driven gesture synthesis increasingly maps speech or text to expressive human or avatar motion. Representative systems model semantic, affective, stylistic, and temporal properties of conversational motion, but commonly assume a shared human-like skeleton and evaluate naturalness or speech synchrony. Social-robot work, in contrast, demonstrates that gesture can communicate semantic information but is usually robot-specific and manually authored. Our setting differs in holding a communicative cue fixed while varying morphology—including humanoid, mobile, and single-arm robots—and in testing whether an LLM output representation preserves the cue's effect rather than reproducing a literal human pose.

**Cite in this paragraph:** Kopp06, Salem12, Chidambaram12, Ginosar19, Yoon20, Gesticulator20, Ahuja20, BEAT22, CaMN22, GENEA20/22.  Add one robot-gesture citation and 4–6 data-driven gesture citations in the final version; do not cite the complete table inline.

### Paragraph 2 — Motion primitives, language grounding, and correction

Movement-primitive research provides compact parameterizations such as DMPs, probabilistic movement primitives, and task-parameterized models for learning, adapting, and reproducing trajectories. Language-conditioned robotics instead commonly maps instructions to a task plan, policy, goal, or low-level action sequence; recent work also grounds language in DMP parameters and uses verbal feedback to alter plans or policies. These approaches establish useful mechanisms for execution and interactive repair, but they generally target object-centric manipulation or assume one fixed primitive/policy interface. We instead make the **language-executable abstraction** the experimental variable, comparing vector demonstrations, symbolic motion programs, and trajectory sketches for their ability to preserve multi-factor communicative motion across heterogeneous embodiments.

**Cite in this paragraph:** Schaal06, Ijspeert13, Paraschos13, Calinon16, Stepputtis20, Nair22, SayCan22, RT-2, LMP26, Sharma22, Broad17, Yell24.  This paragraph is the closest intellectual home for the paper.

## Candidate reference table

Relation depth: **Direct** = should be discussed in prose; **Strong** = cite if space permits; **Context** = use for positioning, comparison, or rebuttal.

| ID | Work | Area | Paragraph | How to introduce it | Relation |
|---|---|---|---|---|---|
| G01 | Kopp et al., *Towards a Common Framework for Multimodal Generation: The Behavior Markup Language* (2006) | Multimodal behavior specification | P1 | Hand-authored symbolic behavior-language baseline for communicative agents | Strong |
| G02 | Salem et al., *Generation and Evaluation of Communicative Robot Gesture* (2012) | Social robot gesture | P1 | Robot-specific communicative gesture generation/evaluation | Direct |
| G03 | Chidambaram et al., *Communicating Semantic Information Through Gesture to Improve Human-Robot Interaction* (2012) | HRI gesture semantics | P1 | Evidence that gesture carries task-relevant semantic information | Direct |
| G04 | Ginosar et al., *Learning Individual Styles of Conversational Gesture* (CVPR 2019) | Data-driven co-speech gesture | P1 | Learned expressive gesture under a shared human body | Strong |
| G05 | Yoon et al., *Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity* (SIGGRAPH 2020) | Speech/text-to-gesture | P1 | Text, audio, and speaker conditioning for human gesture synthesis | Strong |
| G06 | Kucherenko et al., *Analyzing Input and Output Representations for Speech-Driven Gesture Generation* (2020) | Gesture representation | P1 | Closest representation-design precedent in co-speech gesture synthesis | Direct |
| G07 | Kucherenko et al., *Gesticulator: A Framework for Semantically-Aware Speech-Driven Gesture Generation* (2020) | Semantically-aware gesture | P1 | Language semantics in learned human gesture generation | Direct |
| G08 | Ahuja et al., *No Gesture Left Behind: Learning Relationships between Spoken Language and Freeform Gestures* (2020) | Language–gesture alignment | P1 | Language/gesture correspondence without robot embodiment | Strong |
| G09 | Bhattacharya et al., *Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning* (2021) | Affective gesture | P1 | Affective and manner-like semantic factors matter beyond path geometry | Context |
| G10 | Alexanderson et al., *Style-Controllable Speech-Driven Gesture Synthesis* (2020) | Style-controlled gesture | P1 | Style/"how" as an explicit gesture variable | Strong |
| G11 | Yoon et al., *CaMN: Context-aware Motion Network for Co-speech Gesture Generation* (2022) | Context-aware gesture | P1 | Context and temporal composition in speech-driven gesture | Strong |
| G12 | Liu et al., *BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis and Evaluation* (CVPR 2022) | Gesture benchmark | P1 | Large human-gesture benchmark; contrast its shared human skeleton with heterogeneous robots | Direct |
| G13 | Zhu et al., *DiffGesture: Generating Human Gesture from Audio with Diffusion Probabilistic Models* (2023) | Generative gesture | P1 | Modern high-fidelity gesture generation, but not executable robot motion | Context |
| G14 | Liu et al., *EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling* (2024) | Holistic gesture | P1 | Whole-body/hand/face coordination as high-dimensional gesture structure | Strong |
| G15 | Kucherenko et al., *The GENEA Challenge 2020: Benchmarking Gesture Generation* (2021) | Evaluation benchmark | P1 | Human evaluation protocol precedent for generated gestures | Direct |
| G16 | Kucherenko et al., *The GENEA Challenge 2022: A Large Evaluation of Data-Driven Co-Speech Gesture Generation* (2023) | Evaluation benchmark | P1 | Scalable comparative human evaluation for motion generation | Direct |
| G17 | Habibie et al., *Learning Speech-driven 3D Conversational Gesture Generation from Video* (2022) | 3D gesture generation | P1 | Learning 3D conversational motion from real data | Context |
| G18 | Ao et al., *Rhythmic Gesticulator: Rhythm-Aware Co-Speech Gesture Synthesis* (2022) | Temporal gesture structure | P1 | Tempo/rhythm as meaning-bearing motion information | Context |
| M01 | Schaal, *Dynamic Movement Primitives—A Framework for Motor Control in Humans and Humanoid Robotics* (2006) | DMP | P2 | Foundational compact motor-primitive representation | Direct |
| M02 | Ijspeert et al., *Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors* (2013) | DMP | P2 | Canonical DMP formulation and stability motivation | Direct |
| M03 | Pastor et al., *Learning and Generalization of Motor Skills by Learning from Demonstration* (ICRA 2009) | DMP from demonstration | P2 | Demonstration-driven DMP learning/generalization | Strong |
| M04 | Kober and Peters, *Policy Search for Motor Primitives in Robotics* (2011) | Motor primitive learning | P2 | Primitive parameters as a compact optimization space | Strong |
| M05 | Paraschos et al., *Probabilistic Movement Primitives* (NeurIPS 2013) | ProMP | P2 | Distributional primitive representation and conditioning | Direct |
| M06 | Calinon, *A Tutorial on Task-Parameterized Movement Learning and Retrieval* (2016) | TP-GMM | P2 | Task-frame adaptation/retargeting precedent | Direct |
| M07 | Stulp et al., *Learning Movement Primitives with Gaussian Process Regression* (Humanoids 2013) | Learned primitives | P2 | Nonparametric primitive learning | Context |
| M08 | Ude et al., *Task-Specific Generalization of Dynamic Movement Primitives* (T-RO 2014) | DMP generalization | P2 | Generalization of a primitive under changing task constraints | Strong |
| M09 | Calinon et al., *Learning and Reproduction of Gestures by Imitation* (2012) | Gesture primitives | P2 | Imitation-based reproduction of motion/gesture structure | Strong |
| M10 | Kormushev et al., *Imitation Learning of Positional and Force Skills Demonstrated via Kinesthetic Teaching and Haptic Input* (2011) | Skill learning | P2 | Shows the contact-rich regime deliberately excluded here | Context |
| M11 | Noseworthy et al., *Task-Conditioned Variational Autoencoders for Learning Movement Primitives* (CoRL 2020) | Learned primitive latent | P2 | Task-conditioned latent primitive alternative | Strong |
| M12 | Ravichandar et al., *Recent Advances in Robot Learning from Demonstration* (Annual Review 2020) | LfD survey | P2 | Survey/context for demonstration and representation choices | Context |
| L01 | Tellex et al., *Understanding Natural Language Commands for Robotic Navigation and Mobile Manipulation* (AAAI 2011) | Grounded language | P2 | Early language-to-grounded-robot-action formulation | Strong |
| L02 | Matuszek et al., *Learning from Unscripted Deictic Gesture and Language for Human-Robot Interactions* (AAAI 2013) | Gesture + language grounding | P2 | Multimodal grounding of language and human gesture | Strong |
| L03 | Mei et al., *Listen, Attend, and Walk: Neural Mapping of Navigation Instructions to Action Sequences* (AAAI 2016) | Language-to-action | P2 | End-to-end instruction-to-action baseline family | Context |
| L04 | Misra et al., *Mapping Instructions and Visual Observations to Actions with Reinforcement Learning* (EMNLP 2017) | Language-conditioned control | P2 | Language+vision to actions; contrast environment-dependent execution | Context |
| L05 | Anderson et al., *On Evaluation of Embodied Navigation Agents* / Vision-and-Language Navigation (2018) | Embodied language benchmark | P2 | Benchmark precedent, but object/environment-centric | Context |
| L06 | Shridhar et al., *ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks* (CVPR 2020) | Household language tasks | P2 | “Everyday task” language but evaluated through object-state success | Strong |
| L07 | Stepputtis et al., *Language-Conditioned Imitation Learning for Robot Manipulation Tasks* (NeurIPS 2020) | Language + demonstrations | P2 | Direct predecessor for language-conditioned motion from demonstrations | Direct |
| L08 | Nair et al., *Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation* (CoRL 2022) | Language-conditioned behavior | P2 | Offline language grounding/policy learning comparison | Strong |
| L09 | Shridhar et al., *CLIPort: What and Where Pathways for Robotic Manipulation* (CoRL 2021) | Language-conditioned manipulation | P2 | Object-centric language-conditioned manipulation baseline | Context |
| L10 | Jang et al., *BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning* (CoRL 2022) | Generalist robot policy | P2 | Natural-language task conditioning and zero-shot transfer | Context |
| L11 | Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (2022) | SayCan | P2 | Language planning over executable robot skills | Strong |
| L12 | Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale* (2022) | Robot transformer | P2 | Large-scale low-level policy alternative to explicit abstractions | Context |
| L13 | Driess et al., *PaLM-E: An Embodied Multimodal Language Model* (2023) | Embodied VLM | P2 | General embodied language model; perception/task planning outside scope | Context |
| L14 | Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023) | VLA | P2 | Web-to-robot knowledge transfer; contrast direct action-token output | Strong |
| L15 | Liang et al., *Code as Policies: Language Model Programs for Embodied Control* (ICRA 2023) | Programmatic control | P2 | Language-program interface / compositionality precedent | Direct |
| L16 | Huang et al., *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models* (CoRL 2023) | LLM planning | P2 | Structured intermediate representation for language-to-action | Strong |
| L17 | Jiang et al., *VIMA: General Robot Manipulation with Multimodal Prompts* (ICML 2023) | Prompted robot policy | P2 | Multimodal prompt grounding for manipulation | Context |
| L18 | Mees et al., *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks* (RA-L 2022) | Benchmark | P2 | Language-conditioned manipulation benchmark contrast | Strong |
| L19 | Octo Model Team, *Octo: An Open-Source Generalist Robot Policy* (2024) | Generalist policy | P2 | Generalist policy alternative; not a semantic motion abstraction | Context |
| L20 | CLIP-RT, *Learning Language-Conditioned Robotic Policies from Natural Language Supervision* (2024) | Language motion primitives | P2 | Predicts language-based motion primitives; close terminology/positioning | Direct |
| L21 | Dai et al., *Language Movement Primitives: Grounding Language Models in Robot Motion* (2026) | VLM-to-DMP | P2 | Closest recent work: VLM-selected DMPs for tabletop manipulation | Direct |
| C01 | Broad et al., *Real-time Natural Language Corrections for Assistive Robotic Manipulators* (IJRR 2017) | Online correction | P2 | Real-time corrective language modifies robot behavior | Direct |
| C02 | Co-Reyes et al., *Guiding Policies with Language via Meta-Learning* (ICLR 2019) | Iterative corrections | P2 | Policy learns from an instruction plus a sequence of corrections | Direct |
| C03 | Sharma et al., *Correcting Robot Plans with Natural Language Feedback* (RSS 2022) | Plan correction | P2 | Language transformed into cost/goal/constraint changes | Direct |
| C04 | Karamcheti et al., *Shared Autonomy for Robotic Manipulation with Language Corrections* (LILAC, 2023) | Streaming corrections | P2 | Correction-aware language control in shared autonomy | Strong |
| C05 | Shi et al., *Yell At Your Robot: Improving On-the-Fly from Language Corrections* (2024) | Online adaptation | P2 | Fine-grained verbal feedback for policy correction | Direct |
| C06 | ExTraCT, *Explainable Trajectory Corrections for Language-Based Human-Robot Interaction Using Textual Feature Descriptions* (2024) | Trajectory editing | P2 | Modular trajectory modification from language | Strong |
| C07 | *Interactive Robot Learning from Verbal Correction* (2026) | Verbal correction learning | P2 | Trajectory–verbal-correction pairs for improving policies | Strong |

## What each paragraph must and must not claim

| Paragraph | Must establish | Must not claim | Best evidence from this paper |
|---|---|---|---|
| P1: Gesture generation | Gesture meaning is multi-factor and current generation work mainly uses shared human/robot bodies | That human/avatar gesture synthesis directly solves executable cross-robot motion | 30 cues × 5 embodiments; recognition and factor failures |
| P2: Primitives, grounding, correction | Compact intermediate representations enable composition, retargeting, and repair; language robotics often targets task completion | That Motion Program is a DMP or inherits DMP stability guarantees | Interface comparison; token scaling; later correction-locality experiment |

## Recommended final citation budget

Use 16–22 references in the main paper, not all 58.

- **Paragraph 1 (7–9):** Kopp06, Salem12, Chidambaram12, Yoon20, Gesticulator20, Ahuja20, BEAT22, CaMN22, GENEA20/22.
- **Paragraph 2 (9–12):** Schaal06, Ijspeert13, Paraschos13, Calinon16, Stepputtis20, SayCan22, RT-2, Code as Policies, LMP26, Broad17, Sharma22, Yell24.
- **Experiments/limitations (as needed):** ALFRED, CALVIN, CLIPort, VIMA, VoxPoser, RT-1, Octo, LILAC, ExTraCT.

## Primary links for the core set

- [BEAT (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_BEAT_A_Large-Scale_Semantic_and_Emotional_Multi-Modal_Dataset_for_Conversational_Gestures_CVPR_2022_paper.html)
- [Dynamical Movement Primitives (Ijspeert et al., 2013)](https://doi.org/10.1162/NECO_a_00393)
- [Task-Parameterized Movement Learning tutorial (Calinon, 2016)](https://calinon.ch/papers/Calinon-TPGMM-tutorial-2016.pdf)
- [Language-Conditioned Imitation Learning (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/9909794d52985cbc5d95c26e31125d1a-Paper.pdf)
- [LOReL / language-conditioned robot behavior (CoRL 2022)](https://proceedings.mlr.press/v164/nair22a.html)
- [SayCan](https://say-can.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
- [VoxPoser](https://voxposer.github.io/)
- [Language Movement Primitives (2026)](https://arxiv.org/abs/2602.02839)
- [Real-time Natural Language Corrections (Broad et al., 2017)](https://doi.org/10.1177/0278364917706418)
- [Correcting Robot Plans with Natural Language Feedback (RSS 2022)](https://www.roboticsproceedings.org/rss18/p065.pdf)
- [Yell At Your Robot (2024)](https://arxiv.org/abs/2403.12910)

## Immediate writing consequence

The paper should call the proposed output a **language-executable motion abstraction** or **symbolic communicative motion program**, not a “Language Movement Primitive.” The latter overlaps directly with Dai et al. (2026) and suggests DMP-style dynamical/stability guarantees that this work does not claim.
