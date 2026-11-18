# Synthetic Continuum Morphologies (GCR)

Gesture Continuum Robot family: one parametric body plan sampled at three
points along \(\alpha \in [0,1]\). Used to argue that Motion Code attributes
are not catalog-overfit to the five real platforms.

## Prior work we read (synthetic / grammar morphology)

1. **Zhao et al., RoboGrammar (TOG 2020)** — recursive graph grammar → fabricable
   robot graphs; compiler to simulation. Classic “grammar → XML-like body”.
2. **Xu et al., GLSO (2022)** — grammar-guided latent optimization on RoboGrammar
   design spaces (sample-efficient morphology search).
3. **RoboMorph (arXiv 2024)** — LLM proposes grammar designs → compile to MuJoCo
   XML → RL fitness (MJX/Brax).
4. **Evolution + differentiable robots (arXiv 2024)** — outer discrete morphology
   mutations + inner gradient control; continuous behavioral fitness over bodies.
5. **Morphological pretraining / zero-shot evolution (2025)** — universal controller
   evaluates non-differentiable body edits quickly.
6. *(bonus)* **MorphoGen / Debate2Create** — LLM mutates XML genotypes / co-designs
   morphology+reward directly.

**Takeaway for us:** prior work synthesizes bodies for *locomotion/task* fitness.
We reuse the idea of a **parametric / grammar-compilable body**, but evaluate
**communicative recognizability** and attribute availability along a continuum.

## Continuum parameter \(\alpha\)

| \(\alpha\) | File | Communicative channels |
|---:|---|---|
| 0.0 | `xml/gcr_alpha0.xml` | base \(x,y,\theta\) only |
| 0.5 | `xml/gcr_alpha05.xml` | + mast lift + 1-DoF stub |
| 1.0 | `xml/gcr_alpha1.xml` | + 3-DoF arm + gripper |

Continuous geometric knobs (interpolated in `yml/continuum_params.yaml`):
`base_radius`, `mast_height`, `arm_reach`, `n_arm_dof`, `has_gripper`.

Attribute schemas: `yml/motion_code_synthetic_continuum.yaml`.
