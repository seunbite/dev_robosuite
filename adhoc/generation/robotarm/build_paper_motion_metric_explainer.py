#!/usr/bin/env python3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'data/results/evaluation/motion_similarity_3d'
BLACK = '#202020'
BLUE = '#1F6FEB'
BLUE_RGB = (31, 111, 235)
GEN = REPO / 'data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/humanoid/gif/emb_g26_depict_tree.gif'
GT = REPO / 'data/results/render/embodiment_motion_code_gt/humanoid/gif/emb_g26_depict_tree.gif'


def frames(path, fractions=(0.28, 0.72)):
    gif = Image.open(path)
    seq = [np.asarray(f.convert('RGB'))[28:] for f in ImageSequence.Iterator(gif)]
    return [seq[min(len(seq) - 1, round((len(seq) - 1) * q))] for q in fractions]


def show(ax, image):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def robot_mask(image):
    """Approximate the humanoid silhouette while excluding floor and shadow."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Bright, low-saturation links separate cleanly from blue sky and dark floor.
    fg = np.where((hsv[..., 1] < 24) & (hsv[..., 2] > 150), 255, 0).astype(np.uint8)
    spatial = np.zeros_like(fg)
    spatial[int(h * .10):int(h * .88), int(w * .08):int(w * .92)] = 255
    # Below the hips, keep only the central leg corridor so floor shadows are not outlined.
    spatial[int(h * .55):, :] = 0
    spatial[int(h * .55):int(h * .90), int(w * .34):int(w * .66)] = 255
    fg = cv2.bitwise_and(fg, spatial)
    # Include the dark gripper/fingertip pixels only around upper-body bright links.
    nearby = cv2.dilate(fg, np.ones((31, 31), np.uint8), iterations=1)
    fingertip = np.where((hsv[..., 2] < 145) & (nearby > 0), 255, 0).astype(np.uint8)
    fingertip[int(h * .55):] = 0
    fg = cv2.bitwise_or(fg, fingertip)
    # Keep link contours separate at the pelvis instead of bridging them into a box.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    # Reject broad horizontal scene bands; even outstretched arms occupy less width.
    broad_rows = np.count_nonzero(fg, axis=1) > int(w * .58)
    fg[broad_rows, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        cx = x + ww / 2
        if (90 < area < h * w * .12 and abs(cx - w / 2) < w * .34
                and ww < w * .72 and hh < h * .80):
            keep[labels == i] = 255
    return keep


def silhouette_outline(image):
    """Outline only the central humanoid."""
    fg = robot_mask(image)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = fg.shape
    # The waist connector is a small rectangular renderer artifact, not a body contour.
    contours = [c for c in contours if not (
        (lambda x, y, ww, hh: (0.43 * h < y < 0.64 * h and ww > 0.09 * w and hh < 0.16 * h))(*cv2.boundingRect(c))
    )]
    out = image.copy()
    cv2.drawContours(out, contours, -1, BLUE_RGB, 4, cv2.LINE_AA)
    # Hide the renderer's rectangular waist bracket; preserve the actual body image there.
    out[int(.43 * h):int(.57 * h), int(.28 * w):int(.72 * w)] = image[int(.43 * h):int(.57 * h), int(.28 * w):int(.72 * w)]
    return out


def motion_overlay(previous, current):
    """Previous + 0.5 current, outlining only humanoid silhouette change."""
    blended = cv2.addWeighted(previous, 0.50, current, 0.50, 0)
    old_mask, new_mask = robot_mask(previous), robot_mask(current)
    # Only the new silhouette portion: this marks where frame t has moved beyond t−1.
    mask = cv2.bitwise_and(new_mask, cv2.bitwise_not(old_mask))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 35]
    cv2.drawContours(blended, contours, -1, BLUE_RGB, 4, cv2.LINE_AA)
    return blended


def compare_arrow_fig(fig, ax_a, ax_b, label, color=BLACK):
    """Place a clear ↔ arrow + d_* at the visual midpoint between image A and B."""
    from matplotlib.patches import FancyArrowPatch

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb_a = ax_a.images[0].get_window_extent(renderer=renderer)
    bb_b = ax_b.images[0].get_window_extent(renderer=renderer)
    cx = float(inv.transform(((bb_a.x0 + bb_a.x1) * 0.5, 0.0))[0])
    y_hi = float(inv.transform((0.0, bb_a.y0))[1])  # bottom of A image
    y_lo = float(inv.transform((0.0, bb_b.y1))[1])  # top of B image
    y_mid = 0.5 * (y_hi + y_lo)
    # Span most of the A↔B gap so the arrow reads clearly.
    half = 0.42 * (y_hi - y_lo)
    arrow = FancyArrowPatch(
        (cx - 0.010, y_mid - half),
        (cx - 0.010, y_mid + half),
        transform=fig.transFigure,
        arrowstyle='<->',
        mutation_scale=12,
        lw=2.0,
        color=color,
        clip_on=False,
    )
    fig.add_artist(arrow)
    fig.text(
        cx + 0.002,
        y_mid,
        label,
        ha='left',
        va='center',
        fontsize=11,
        color=color,
    )


def seam_row_label(fig, ax_left, ax_right, label):
    """Center A/B in the narrow seam between the two metric panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    left = ax_left.images[0].get_window_extent(renderer=renderer)
    right = ax_right.images[0].get_window_extent(renderer=renderer)
    x = float(inv.transform(((left.x1 + right.x0) * 0.5, 0.0))[0])
    y = float(inv.transform((0.0, (left.y0 + left.y1) * 0.5))[1])
    fig.text(x, y, label, ha='center', va='center', fontsize=16, color=BLACK)


def main():
    a, b = frames(GEN), frames(GT)
    # Shared mid-pose makes the aligned-frame example visually unambiguous.
    b[1] = a[1].copy()
    # Keep frames at their native aspect ratio; reduce the canvas, not the image geometry.
    # A compact, high aspect ratio keeps the two metric pairs readable in a
    # single paper column: Pose f1/f2, a small separator, Velocity f1/f2.
    fig = plt.figure(figsize=(7.2, 5.35), facecolor='white')
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.12)

    # Pose: A / B.  Metric titles are positioned from the actual image bounds
    # after layout, so they nearly touch the first row without overlapping it.
    pose = outer[0].subgridspec(2, 2, hspace=-0.10, wspace=0.0)
    pose_pairs = []
    for i in range(2):
        ax_a = fig.add_subplot(pose[0, i]); show(ax_a, silhouette_outline(a[i]))
        if i == 0:
            ax_a.text(-0.04, 0.5, 'A', transform=ax_a.transAxes,
                      ha='right', va='center', fontsize=13, color=BLACK)
        ax_b = fig.add_subplot(pose[1, i]); show(ax_b, silhouette_outline(b[i]))
        if i == 0:
            ax_b.text(-0.04, 0.5, 'B', transform=ax_b.transAxes,
                      ha='right', va='center', fontsize=13, color=BLACK)
        ax_b.set_xlabel(f'frame {i + 1}', fontsize=9, labelpad=3)
        pose_pairs.append((ax_a, ax_b))

    # Velocity: same compact layout.  Put its A/B labels at the left of its own
    # frame pair so the visual reading order is A--f1--f2, then A--f1--f2.
    vel = outer[1].subgridspec(2, 2, hspace=-0.10, wspace=0.0)
    shown_a = [a[0], motion_overlay(a[0], a[1])]
    shown_b = [b[0], motion_overlay(b[0], b[1])]
    labels = ['frame 1', 'frame 2']
    vel_pairs = []
    for i in range(2):
        top = fig.add_subplot(vel[0, i]); show(top, shown_a[i])
        bottom = fig.add_subplot(vel[1, i]); show(bottom, shown_b[i])
        if i == 0:
            top.text(-0.04, 0.5, 'A', transform=top.transAxes,
                     ha='right', va='center', fontsize=10)
            bottom.text(-0.04, 0.5, 'B', transform=bottom.transAxes,
                        ha='right', va='center', fontsize=10)
        bottom.set_xlabel(labels[i], fontsize=9, labelpad=3)
        vel_pairs.append((top, bottom))

    for ax_a, ax_b in pose_pairs:
        compare_arrow_fig(fig, ax_a, ax_b, r'$d_{pose}$', BLACK)
    for ax_a, ax_b in vel_pairs:
        # Same black as d_pose for the label + arrow.
        compare_arrow_fig(fig, ax_a, ax_b, r'$d_{vel}$', BLACK)

    fig.canvas.draw()
    for pair, label in ((pose_pairs, 'Pose-DTW'), (vel_pairs, 'Velocity-DTW')):
        left = pair[0][0].get_position()
        right = pair[-1][0].get_position()
        fig.text(
            0.5 * (left.x0 + right.x1),
            max(p[0].get_position().y1 for p in pair) + 0.002,
            label,
            ha='center', va='bottom', fontsize=14, color=BLACK,
        )

    fig.savefig(OUT / 'paper_motion_metric_explainer.png', dpi=220, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    fig.savefig(OUT / 'paper_motion_metric_explainer.svg', bbox_inches='tight', pad_inches=0.04, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
