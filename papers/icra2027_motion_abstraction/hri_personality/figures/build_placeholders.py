#!/usr/bin/env python3
"""Placeholder figures for HRI personality paper. Dummy numbers encode the intended story."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

C_BLUE = "#2F6DB3"
C_ORANGE = "#D07A2C"
C_GREEN = "#3A8A5A"
C_RED = "#C44B4B"
C_GRAY = "#6B7280"
C_PURPLE = "#6B5B95"
# Exact opaque RGB used by the trajectory in root.pdf Fig. 7:
# adhoc/generation/robotarm/build_llm_robot_distillation_teaser.py
FIG7_BLUE = "#2F72FE"
PAL = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig1_concept():
    """Paper concept: intelligence models context; personality deforms that model."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.65))

    # (a) A visual analogy for effective contextual capacity. The richer function is
    # deliberately smooth: intelligence means sensitivity to relevant variables,
    # not arbitrary oscillation.
    ax = axes[0]
    x = np.linspace(0, 1, 240)
    f_low = 0.18 + 0.62 * x
    f_rich = 0.23 + 0.46 * x + 0.12 * np.sin(3.2 * np.pi * x) * (0.25 + x)
    ax.plot(x, f_low, color=C_GRAY, lw=2.5, label=r"limited model  $f(x_1)$")
    ax.plot(x, f_rich, color=C_BLUE, lw=2.7, label=r"context-rich model  $f(x_1,\ldots,x_d)$")
    ax.fill_between(x, f_low, f_rich, color=C_BLUE, alpha=0.08)
    for xpos, label in [(0.16, "relationship"), (0.48, "group"), (0.79, "urgency")]:
        ax.axvline(xpos, color="#D8DDE5", lw=0.7, zorder=0)
        ax.text(xpos, 0.04, label, rotation=90, va="bottom", ha="center", fontsize=6.8, color=C_GRAY)
    ax.set_xlabel("Context projection  $x$")
    ax.set_ylabel("Normative action tendency  $f(x)$")
    ax.set_ylim(0, 1.03)
    ax.set_title("(a) Social intelligence: model more of context", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # (b) Requested extravert/reserved example. Both policies approach the same
    # norm when there is nobody to address or the crowd becomes constraining.
    ax = axes[1]
    n = np.linspace(0, 30, 241)
    p0 = 0.27 + 0.20 * (1 - np.exp(-n / 7.0)) - 0.18 / (1 + np.exp(-(n - 24) / 2.2))
    envelope = (1 - np.exp(-n / 2.2)) * (1 / (1 + np.exp((n - 25) / 2.0)))
    b = 0.27 * np.tanh((n - 7) / 5.2) * envelope
    ext = np.clip(p0 + b, 0.03, 0.97)
    reserved = np.clip(p0 - b, 0.03, 0.97)
    ax.plot(n, p0, color=C_GRAY, lw=2.0, ls="--", label=r"norm  $f(x)$")
    ax.plot(n, ext, color=C_BLUE, lw=2.7, label=r"extravert  $f(x)+b_{\mathrm{E}}(x)$")
    ax.plot(n, reserved, color=C_ORANGE, lw=2.7, label=r"introverted  $f(x)+b_{\mathrm{I}}(x)$")
    ax.axvspan(0, 1.8, color="#ECEFF3", alpha=0.8)
    ax.axvspan(25, 30, color="#ECEFF3", alpha=0.8)
    ax.annotate("convergence:\nno audience", xy=(0.8, p0[6]), xytext=(4.2, 0.82),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GRAY),
                fontsize=7, color=C_GRAY)
    ax.annotate("suppression under a\nstrongly constraining crowd", xy=(27.5, p0[-21]),
                xytext=(17.0, 0.10), arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GRAY),
                fontsize=7, color=C_GRAY)
    ax.set_xlabel("Number of people present  $x$")
    ax.set_ylabel(r"$P(\mathtt{initiate}\mid x)$")
    ax.set_ylim(0, 1.03)
    ax.set_title("(b) Personality: stable contextual deformation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig1_concept")


def box(ax, x, y, w, h, text, fc="#F3F6FA", ec=C_BLUE, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def fig2_mapping():
    """Forward realization and inverse inference over context/action spaces."""
    rng = np.random.default_rng(23)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.45))
    fs = 10.2
    given_gray = "#737B87"
    light_gray = "#C7CCD4"
    iconic_blue = FIG7_BLUE

    # Shared normalized coordinates within each panel.
    left_center, right_center, radius = (0.23, 0.48), (0.77, 0.48), 0.205
    ctx = np.array([
        [-.12, .10], [-.08, .03], [-.13, -.08], [-.04, -.13], [.03, .14],
        [.09, .07], [.13, -.04], [.07, -.13], [-.01, .00], [.01, -.08],
        [-.16, -.01], [.15, .13], [-.06, .16], [.15, .02], [-.13, .15],
        [.11, -.16], [-.01, -.17], [.05, .05],
    ])
    act = np.array([
        [-.12, .10], [-.11, -.08], [-.02, .15], [.03, .02], [.10, .12],
        [.13, -.05], [.04, -.14], [-.04, -.05], [.14, .04], [-.13, .00],
    ])
    chosen_ctx = [14, 8, 15]
    chosen_act = [2, 4, 6]
    alternative_act = [4, 6, 2]  # same selected actions, different composition

    def P(center, delta):
        return center[0] + delta[0], center[1] + delta[1]

    def curved_edge(ax, p0, p1, color, lw=1.7, alpha=1.0, rad=0.0, ls="-", z=3):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=7,
            connectionstyle=f"arc3,rad={rad}", linewidth=lw, linestyle=ls,
            color=color, alpha=alpha, zorder=z,
        ))

    def draw_spaces(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Circle(left_center, radius, facecolor="#F6F7F8",
                            edgecolor="#9AA1AB", linewidth=1.35, zorder=0))
        ax.add_patch(Circle(right_center, radius, facecolor="#F6F7F8",
                            edgecolor="#9AA1AB", linewidth=1.35, zorder=0))
        ax.scatter(left_center[0] + ctx[:, 0], left_center[1] + ctx[:, 1],
                   s=9, c="#B8BEC8", alpha=0.85, edgecolors="none", zorder=2)
        ax.scatter(right_center[0] + act[:, 0], right_center[1] + act[:, 1],
                   s=11, c="#B8BEC8", alpha=0.85, edgecolors="none", zorder=2)
        ax.text(left_center[0], 0.205, r"context space  $\mathcal{X}$",
                ha="center", fontsize=fs, fontweight="bold")
        ax.text(left_center[0], 0.165, "many social situations",
                ha="center", fontsize=fs, color=C_GRAY)
        ax.text(right_center[0], 0.205, r"action space  $\mathcal{A}$",
                ha="center", fontsize=fs, fontweight="bold")
        ax.text(right_center[0], 0.165, "limited robot primitives",
                ha="center", fontsize=fs, color=C_GRAY)

    # (a) A desired latent personality selects and composes relations.
    ax = axes[0]
    draw_spaces(ax)
    ax.text(0.0, 0.97, "(a) Personality realization", ha="left", va="top",
            fontsize=fs, fontweight="bold")
    box(ax, 0.335, 0.775, 0.33, 0.115,
        r"target latent mapping" + "\n" + r"$p_\theta(a\mid x)$",
        "#F2F3F5", given_gray, fs=fs)
    ax.annotate("", xy=(0.50, 0.70), xytext=(0.50, 0.775),
                arrowprops=dict(arrowstyle="-|>", color=given_gray, lw=1.15))

    # Faint alternatives show the large composition space.
    for _ in range(18):
        i = int(rng.integers(0, len(ctx)))
        j = int(rng.integers(0, len(act)))
        curved_edge(ax, P(left_center, ctx[i]), P(right_center, act[j]),
                    light_gray, lw=0.55, alpha=0.24,
                    rad=float(rng.uniform(-0.13, 0.13)), z=1)
    # A counterfactual policy uses exactly the same highlighted actions but rewires them.
    for i, j in zip(chosen_ctx, alternative_act):
        curved_edge(ax, P(left_center, ctx[i]), P(right_center, act[j]),
                    given_gray, lw=0.75, alpha=0.38, rad=-0.08, ls="--", z=2)
    for k, (i, j) in enumerate(zip(chosen_ctx, chosen_act)):
        p0, p1 = P(left_center, ctx[i]), P(right_center, act[j])
        ax.scatter(*p0, s=42, c=iconic_blue, edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter(*p1, s=42, c=iconic_blue, edgecolors="white", linewidths=0.8, zorder=5)
        curved_edge(ax, p0, p1, iconic_blue, lw=1.75, alpha=0.98,
                    rad=(k - 1) * 0.035, z=4)
    ax.text(0.50, 0.08, r"selected relations compose a signature  $\mathcal{S}_\theta$",
            ha="center", fontsize=fs, color=iconic_blue)

    # (b) Sparse observed relations support inverse recovery of the latent mapping.
    ax = axes[1]
    draw_spaces(ax)
    ax.text(0.0, 0.97, "(b) Personality inference", ha="left", va="top",
            fontsize=fs, fontweight="bold")
    for k, (i, j) in enumerate(zip(chosen_ctx, chosen_act)):
        p0, p1 = P(left_center, ctx[i]), P(right_center, act[j])
        ax.scatter(*p0, s=42, c=given_gray, edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter(*p1, s=42, c=given_gray, edgecolors="white", linewidths=0.8, zorder=5)
        curved_edge(ax, p0, p1, given_gray, lw=1.55, alpha=0.92,
                    rad=(k - 1) * 0.035, z=4)
    box(ax, 0.335, 0.775, 0.33, 0.115,
        r"recovered latent mapping" + "\n" + r"$\hat p_\theta(a\mid x)$",
        "#EDF4FC", iconic_blue, fs=fs)
    # Evidence arrows rise from the relation bundle to the inferred mapping.
    ax.annotate("", xy=(0.50, 0.775), xytext=(0.50, 0.695),
                arrowprops=dict(arrowstyle="-|>", color=iconic_blue, lw=1.15))
    ax.text(0.50, 0.08, r"observer infers $\hat p_\theta$ from sparse $\{(x_t,a_t)\}_{t=1}^{T}$",
            ha="center", fontsize=fs, color=iconic_blue)

    fig.tight_layout(w_pad=1.1)
    save(fig, "fig2_mapping")


def fig3_human_llm():
    """Required human–LLM quantitative and qualitative comparison."""
    fig = plt.figure(figsize=(10.8, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.48, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    T = np.array([1, 2, 4, 8, 16])
    curves = [
        ("Persona prompt", C_ORANGE, "o--", [0.29, 0.33, 0.38, 0.41, 0.44]),
        ("Compiler (ours)", C_BLUE, "s-", [0.31, 0.42, 0.55, 0.68, 0.74]),
        ("Human policy", C_GREEN, "D-", [0.33, 0.46, 0.60, 0.74, 0.79]),
    ]
    for label, col, fmt, vals in curves:
        ax.plot(T, vals, fmt, color=col, ms=5.5, lw=1.8, label=label)
    ax.axhline(0.25, color=C_GRAY, ls=":", lw=0.9, label="4-way chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T)
    ax.set_xticklabels(T)
    ax.set_ylim(0.18, 0.88)
    ax.set_xlabel("Observed context–action episodes  $T$")
    ax.set_ylabel("Personality identification accuracy")
    ax.set_title("(a) Identity accumulates across contexts", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    metrics = ["Local fit", "Cross-context\ncoherence", "Identity", "Control", "Held-out"]
    prompt = [4.21, 2.88, 3.28, 2.41, 3.05]
    compiler = [4.12, 4.01, 4.08, 4.22, 3.86]
    human = [4.38, 4.27, 4.44, 4.15, 4.09]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, prompt, w, color=C_ORANGE, label="Prompt")
    ax.bar(x, compiler, w, color=C_BLUE, label="Compiler")
    ax.bar(x + w, human, w, color=C_GREEN, label="Human")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.2)
    ax.set_ylim(1, 5.25)
    ax.set_ylabel("Mean rating (1–5)")
    ax.set_title("(b) Human–LLM realization gap", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")

    examples = [
        ("UNSEEN GROUP", "Extravert", "approach + initiate", "Human ✓  Compiler ✓\nPrompt: wait"),
        ("BUSY USER", "Reserved", "wait + respond only", "Human ✓  Compiler ✓\nPrompt: initiate"),
        ("EMERGENCY", "Both personas", "help immediately", "Human ✓  Compiler ✓\nRigid: talks"),
        ("COSTLY HELP", "Helpful", "inspect, then help", "Human ✓  Compiler: help\nPrompt: continue"),
    ]
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    for j, (ctx, target, action, note) in enumerate(examples):
        col = [C_BLUE, C_ORANGE, C_RED, C_GREEN][j]
        ax.add_patch(FancyBboxPatch((j + 0.03, 0.08), 0.92, 0.82,
                                    boxstyle="round,pad=0.02,rounding_size=0.035",
                                    facecolor="#FAFAFA", edgecolor=col, linewidth=1.2))
        ax.text(j + 0.49, 0.82, ctx, ha="center", va="top", fontsize=7.4,
                fontweight="bold", color=col)
        ax.text(j + 0.49, 0.61, target, ha="center", fontsize=7.2)
        ax.text(j + 0.49, 0.43, action, ha="center", fontsize=7.2, fontweight="bold")
        ax.text(j + 0.49, 0.14, note, ha="center", fontsize=6.2, color="#374151")
    ax.set_title("(c) Qualitative policy cases (replace with robot stills and participant quotes)",
                 loc="left", fontsize=9, fontweight="bold", pad=2)
    fig.text(0.995, 0.005, "ALL VALUES ARE DESIGN PLACEHOLDERS", ha="right",
             fontsize=6.5, color=C_GRAY)
    save(fig, "fig3_human_llm")


def fig4_llm_capability():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45))

    # A: human vs LLM cost ratings
    ax = axes[0]
    n = 80
    h = rng.uniform(1, 7, n)
    llm = 0.72 * h + rng.normal(0, 0.9, n) + 0.8
    ax.scatter(h, llm, s=18, c=C_BLUE, alpha=0.7, edgecolors="none")
    ax.plot([1, 7], [1, 7], color=C_GRAY, ls="--", lw=0.9)
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xlabel("Human cost rating")
    ax.set_ylabel("LLM cost rating")
    ax.set_title(r"Cost agreement  $r_s{=}0.71$")
    ax.text(1.2, 6.5, "PLACEHOLDER", fontsize=7, color=C_GRAY)

    # B: diagnostic context ranking
    ax = axes[1]
    methods = ["Random\ncontexts", "LLM\ndiagnostic", "Human-\nchosen"]
    acc = [0.34, 0.61, 0.68]
    err = [0.04, 0.03, 0.03]
    bars = ax.bar(methods, acc, yerr=err, color=[C_GRAY, C_ORANGE, C_BLUE], capsize=3, width=0.62)
    ax.axhline(0.25, color="#999", ls=":", lw=0.8)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("4-way personality ID acc.")
    ax.set_title("Diagnostic context generation")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=8)

    # C: KL vs situational pressure
    ax = axes[2]
    pressure = np.linspace(0.1, 0.95, 8)
    kl_prompt = 0.55 - 0.12 * pressure + rng.normal(0, 0.03, 8)
    kl_comp = 0.62 * (1 - pressure) ** 1.15 + 0.08
    ax.plot(pressure, np.clip(kl_prompt, 0.15, 1), "o-", color=C_ORANGE, label="Persona prompt", ms=5)
    ax.plot(pressure, kl_comp, "s-", color=C_BLUE, label="Compiler (ours)", ms=5)
    ax.set_xlabel("Situational pressure  P(x)")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p_i \| p_0)$")
    ax.set_title("Personality suppression")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_llm_capability")


def fig5_inference():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
    T = np.array([1, 2, 4, 8, 16])
    acc_prompt = np.array([0.29, 0.33, 0.38, 0.41, 0.44])
    acc_comp = np.array([0.31, 0.42, 0.55, 0.68, 0.74])
    acc_human = np.array([0.33, 0.46, 0.60, 0.74, 0.79])
    chance = 0.25
    ax = axes[0]
    ax.plot(T, acc_prompt, "o--", color=C_ORANGE, label="LLM persona prompt", ms=6)
    ax.plot(T, acc_comp, "s-", color=C_BLUE, label="LLM compiler (ours)", ms=6)
    ax.plot(T, acc_human, "D-", color=C_GREEN, label="Human-authored policy", ms=6)
    ax.axhline(chance, color=C_GRAY, ls=":", label="Chance (4-way)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T)
    ax.set_xticklabels(T)
    ax.set_ylim(0.15, 0.95)
    ax.set_xlabel("Observed episodes  T")
    ax.set_ylabel("Personality identifiability")
    ax.set_title("Study 2a: evidence accumulation")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    labels = ["Actions only\n(no context)", "Contexts +\nactions", "Marginal-\nmatched + ctx"]
    x = np.arange(3)
    w = 0.36
    extra_vs_res = [0.28, 0.71, 0.66]
    help_vs_self = [0.31, 0.64, 0.58]
    ax.bar(x - w / 2, extra_vs_res, w, color=C_BLUE, label="Extravert vs reserved")
    ax.bar(x + w / 2, help_vs_self, w, color=C_ORANGE, label="Helpful vs self-prior.")
    ax.axhline(0.5, color=C_GRAY, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Forced-choice accuracy")
    ax.set_title("Study 2b: matched marginals")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_inference")


def fig6_realization():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    metrics = ["Local\nappropriateness", "Cross-context\nconsistency", "Identifiability\n(T=8)", "Controllability\n(swap θ)", "Unseen-ctx\ngeneralization"]
    prompt = [4.21, 2.88, 3.28, 2.41, 3.05]
    compiler = [4.12, 4.01, 4.08, 4.22, 3.86]
    human = [4.38, 4.27, 4.44, 4.15, 4.09]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, prompt, w, color=C_ORANGE, label="Persona prompt LLM")
    ax.bar(x, compiler, w, color=C_BLUE, label="Personality compiler")
    ax.bar(x + w, human, w, color=C_GREEN, label="Human policy")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(1, 5.2)
    ax.set_ylabel("Mean rating / scaled score (1–5)")
    ax.set_title("Study 2c: realization quality (placeholder means, n≈48 raters)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.axhline(3, color="#ddd", lw=0.6)
    fig.tight_layout()
    save(fig, "fig6_realization")


def fig6_robot_qual():
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 4.8))
    episodes = [
        ("Extravert", "Lobby, 8 people", "approach +\ninitiate"),
        ("Reserved", "Lobby, 8 people", "wait +\nrespond only"),
        ("Helpful", "Object dropped\nnear cart", "inspect then\nhelp"),
        ("Cautious", "Object dropped\nnear cart", "wait / inspect\nfirst"),
        ("Extravert", "Do-not-disturb", "continue own\ntask (suppress)"),
        ("Reserved", "Do-not-disturb", "continue own\ntask (shared)"),
        ("Adaptive", "Emergency", "normative\nhelp path"),
        ("Rigid extravert", "Emergency", "initiate talk\n(inappropriate)"),
    ]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_BLUE, C_ORANGE, C_GREEN, C_RED]
    for ax, (p, ctx, act), col in zip(axes.ravel(), episodes, colors):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.6)
        ax.add_patch(Rectangle((0.12, 0.28), 0.76, 0.52, facecolor="#F4F4F4", edgecolor="#CCCCCC"))
        ax.text(0.5, 0.54, "[robot still]", ha="center", va="center", fontsize=9, color=C_GRAY)
        ax.text(0.5, 0.38, act, ha="center", va="center", fontsize=7.5, color="#222")
        ax.set_title(f"{p}\n{ctx}", fontsize=7.5, color=col, pad=4)
    fig.suptitle("Placeholder qualitative grid — replace with in-lab stills / video frames", fontsize=10)
    fig.tight_layout()
    save(fig, "fig6_robot_qual")


def fig7_robot_study():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.55))
    metrics = ["Personality\nrecognizability", "Appropriateness", "Perceived\nintelligence", "Likeability", "Trust"]
    rigid = [4.12, 3.21, 3.18, 3.34, 3.28]
    adaptive = [3.98, 4.26, 4.11, 4.02, 3.91]
    prompt = [3.05, 3.72, 3.40, 3.51, 3.22]
    x = np.arange(len(metrics))
    w = 0.25
    ax = axes[0]
    ax.bar(x - w, prompt, w, color=C_ORANGE, label="Prompt-only robot")
    ax.bar(x, rigid, w, color=C_RED, label="Rigid signature")
    ax.bar(x + w, adaptive, w, color=C_BLUE, label="Adaptive (ours)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(1, 5.3)
    ax.set_ylabel("Mean (7-pt scaled to 1–5 placeholder)")
    ax.set_title("Study 3: in-person HRI ratings")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    # scatter appropriateness vs recognizability per participant means
    rng = np.random.default_rng(11)
    for name, col, mx, my in [
        ("Prompt", C_ORANGE, 3.05, 3.72),
        ("Rigid", C_RED, 4.12, 3.21),
        ("Adaptive", C_BLUE, 3.98, 4.26),
    ]:
        xs = rng.normal(mx, 0.22, 18)
        ys = rng.normal(my, 0.22, 18)
        ax.scatter(xs, ys, s=22, c=col, alpha=0.55, label=name, edgecolors="none")
        ax.scatter([mx], [my], s=70, c=col, marker="D", edgecolors="k", linewidths=0.4)
    ax.set_xlabel("Personality recognizability")
    ax.set_ylabel("Situational appropriateness")
    ax.set_xlim(2.2, 5.0)
    ax.set_ylim(2.2, 5.0)
    ax.set_title("Identity–appropriateness tradeoff")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig7_robot_ratings")


def fig9_cycle():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    labels = ["Prompt", "Compiler", "Human"]
    recover = np.array([
        [0.41, 0.18, 0.12],
        [0.14, 0.68, 0.09],
        [0.11, 0.10, 0.74],
    ])
    im = ax.imshow(recover, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inferred personality source")
    ax.set_ylabel("Generated by")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{recover[i, j]:.2f}", ha="center", va="center",
                    color="white" if recover[i, j] > 0.45 else "#111", fontsize=10)
    ax.set_title("Cycle consistency  generate → infer  (placeholder)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(correct identity)")
    fig.tight_layout()
    save(fig, "fig9_cycle")


if __name__ == "__main__":
    fig1_concept()
    fig2_mapping()
    fig3_human_llm()
    fig4_llm_capability()
    fig5_inference()
    fig6_robot_qual()
    fig7_robot_study()
    print("wrote", list(OUT.glob("fig*.png")))
