"""Shared Motion Code speed → duration / frame-count (Roomba reference).

Design goals (render-only; no re-inference):
1. Same ``spd`` semantics on every embodiment.
2. ``spd`` clamped to [SPD_MIN, SPD_MAX]; duration ∝ 1/spd (linear velocity).
3. Path timing uses **path length** — line stroke and arc ``r·θ`` share one
   linear speed so a diameter ``2r`` and an arc of length ``2r`` take equal time.

No min/max frame clamp — raw ``round(t * fps * FRAME_SCALE)``.
``MOTION_FRAME_SCALE`` env overrides density (default **0.5**).

Reference (Roomba): at ``spd=1``, traverse ``V_REF_M_PER_S`` meters per second.
In-place yaw (r≈0) uses the legacy Roomba unit ``90° ≡ V_REF`` meters.
"""
from __future__ import annotations

import math
import os

# Roomba line unit / shared linear speed at spd=1.
V_REF_M_PER_S = 0.15
SPD_MIN = 0.25
SPD_MAX = 4.0
# Joint / oscillating "beat": one rep at spd=1 lasts this many seconds.
BEAT_S_AT_SPD_1 = 1.0
# Default density after A/B probe (0.5× felt best across embodiments).
_DEFAULT_FRAME_SCALE = 0.5


def _frame_scale() -> float:
    """Read live so A/B renders can change ``MOTION_FRAME_SCALE`` mid-process."""
    try:
        return float(os.environ.get("MOTION_FRAME_SCALE", str(_DEFAULT_FRAME_SCALE)))
    except ValueError:
        return _DEFAULT_FRAME_SCALE


# Back-compat alias (import-time snapshot); prefer ``_frame_scale()``.
FRAME_SCALE = _frame_scale()


def clamp_speed(speed: float) -> float:
    return float(min(SPD_MAX, max(SPD_MIN, float(speed))))


# Real-robot / DROID cap: stretch time so authored ``spd`` never exceeds these.
# Never speeds motion up. Override with MOTION_MAX_EE_M_S / MOTION_MAX_QDOT_DEG_S.
_DEFAULT_MAX_EE_M_S = 0.08
_DEFAULT_MAX_QDOT_DEG_S = 30.0


def max_ee_m_s() -> float:
    try:
        return max(0.02, float(os.environ.get("MOTION_MAX_EE_M_S", str(_DEFAULT_MAX_EE_M_S))))
    except ValueError:
        return _DEFAULT_MAX_EE_M_S


def max_qdot_rad_s() -> float:
    try:
        deg = float(os.environ.get("MOTION_MAX_QDOT_DEG_S", str(_DEFAULT_MAX_QDOT_DEG_S)))
    except ValueError:
        deg = _DEFAULT_MAX_QDOT_DEG_S
    return math.radians(max(10.0, deg))


def stretch_duration(
    authored_s: float,
    *,
    ee_travel_m: float = 0.0,
    max_abs_dq_rad: float = 0.0,
) -> tuple[float, str]:
    """Return ``(duration_s, reason)``. ``reason`` is empty if no stretch.

    Caps Cartesian speed and peak |Δq|/Δt. Used by manipulator GIF + npz.
    """
    t = max(0.0, float(authored_s))
    reasons: list[str] = []
    ee_cap = max_ee_m_s()
    q_cap = max_qdot_rad_s()
    travel = abs(float(ee_travel_m))
    dq = abs(float(max_abs_dq_rad))
    if travel > 1e-6:
        need = travel / ee_cap
        if need > t + 1e-9:
            reasons.append(f"EE {travel:.3f}m @{ee_cap:.2f}m/s")
            t = need
    if dq > 1e-9:
        need = dq / q_cap
        if need > t + 1e-9:
            reasons.append(f"|Δq| {math.degrees(dq):.1f}° @{math.degrees(q_cap):.0f}°/s")
            t = need
    return t, "; ".join(reasons)


def frame_bounds(fps: int) -> tuple[int, int]:
    """Deprecated no-op bounds (kept for call-site compat). No clamping."""
    del fps
    return 1, 10**9


def path_length_meters(
    *,
    stroke_m: float = 0.0,
    sweep_deg: float = 0.0,
    radius_m: float = 0.0,
) -> float:
    """One-way path length in meters (caller multiplies by rep when needed)."""
    if abs(float(sweep_deg)) > 1e-6:
        r = abs(float(radius_m))
        deg = abs(float(sweep_deg))
        if r > 1e-5:
            return r * math.radians(deg)
        # In-place spin: 90° @ spd=1 → 1s ≡ V_REF meters.
        return (deg / 90.0) * V_REF_M_PER_S
    return abs(float(stroke_m))


def duration_seconds(
    *,
    speed: float,
    stroke_m: float = 0.0,
    sweep_deg: float = 0.0,
    radius_m: float = 0.0,
    oscillating: bool = False,
    rep: int = 1,
) -> float:
    """Wall-clock seconds for a path or oscillating cycle set.

    - Path with length (line / arc): ``t = (rep · length) / (spd · V_ref)``.
      ``spd`` sets linear speed; ``rep`` only repeats — ``rep=3`` ⇒ 3× time.
    - Oscillating **without** length (joint beats): ``t = rep / spd`` seconds.
    - Oscillating **with** ``stroke_m`` / arc length: treat ``stroke_m`` as the
      length of **one** cycle (e.g. ±0.15 m ⇒ cycle ``2·0.30 = 0.60`` m); total
      ``rep · stroke_m`` at the same linear speed.
    """
    spd = clamp_speed(speed)
    reps = max(1, int(rep))
    length = path_length_meters(
        stroke_m=stroke_m, sweep_deg=sweep_deg, radius_m=radius_m
    )
    if oscillating and length < 1e-6:
        # Joint / head beats — no authored path length.
        return (reps / spd) * BEAT_S_AT_SPD_1
    if length < 1e-6:
        return 0.0
    # One-way: length is the full authored stroke (caller may already fold rep).
    # Oscillating + length: length is one cycle; multiply by rep.
    total = length * reps if oscillating else length
    return total / (spd * V_REF_M_PER_S)


def path_n_frames(
    *,
    fps: int,
    speed: float,
    oscillating: bool = False,
    rep: int = 1,
    stroke_m: float = 0.0,
    sweep_deg: float = 0.0,
    radius_m: float = 0.0,
) -> int:
    """Frame count for a path stroke (0 if zero-length one-way). No clamp."""
    t = duration_seconds(
        speed=speed,
        stroke_m=stroke_m,
        sweep_deg=sweep_deg,
        radius_m=radius_m,
        oscillating=oscillating,
        rep=rep,
    )
    if t <= 0.0 and not oscillating:
        return 0
    n = int(round(t * max(1, int(fps)) * _frame_scale()))
    return max(1, n)


def beat_n_frames(*, fps: int, speed: float, rep: int = 1) -> int:
    """Joint-move / beat timing: ``rep/spd`` seconds (no clamp)."""
    return path_n_frames(
        fps=fps, speed=speed, oscillating=True, rep=max(1, int(rep))
    )
