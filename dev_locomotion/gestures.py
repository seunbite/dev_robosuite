from __future__ import annotations

from dataclasses import dataclass

from .commands import FootTargetMap
from .config import LEG_ORDER
from .kinematics import QuadrupedKinematics, lerp_vec


def offset_feet(feet: FootTargetMap, **leg_offsets: tuple[float, float, float]) -> FootTargetMap:
    updated = dict(feet)
    for leg, offset in leg_offsets.items():
        src = updated[leg]
        updated[leg] = (src[0] + offset[0], src[1] + offset[1], src[2] + offset[2])
    return updated


@dataclass(frozen=True)
class PoseFrame:
    duration: float
    feet: FootTargetMap


@dataclass(frozen=True)
class GestureSequence:
    name: str
    frames: tuple[PoseFrame, ...]
    loop: bool = False

    def sample(self, t: float) -> FootTargetMap:
        if not self.frames:
            raise ValueError(f"Gesture {self.name} has no frames")

        if len(self.frames) == 1:
            return self.frames[0].feet

        total_duration = sum(frame.duration for frame in self.frames[:-1])
        if self.loop and total_duration > 0.0:
            t = t % total_duration
        elif t >= total_duration:
            return self.frames[-1].feet

        elapsed = 0.0
        for current, nxt in zip(self.frames[:-1], self.frames[1:], strict=True):
            segment = current.duration
            if t <= elapsed + segment or nxt is self.frames[-1]:
                alpha = 0.0 if segment <= 0.0 else (t - elapsed) / segment
                return {
                    leg: lerp_vec(current.feet[leg], nxt.feet[leg], alpha)
                    for leg in LEG_ORDER
                }
            elapsed += segment

        return self.frames[-1].feet


class GestureLibrary:
    def __init__(self, kinematics: QuadrupedKinematics) -> None:
        self._kinematics = kinematics
        self._poses = self._build_pose_library()
        self._sequences = self._build_sequence_library()

    def pose(self, name: str) -> FootTargetMap:
        return dict(self._poses[name])

    def sequence(self, name: str) -> GestureSequence:
        return self._sequences[name]

    @property
    def pose_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._poses))

    @property
    def sequence_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._sequences))

    def _build_pose_library(self) -> dict[str, FootTargetMap]:
        stand = self._kinematics.nominal_stance()
        crouch = {leg: (x, y, z + 0.06) for leg, (x, y, z) in stand.items()}
        bow = {
            "FR": (stand["FR"][0] + 0.01, stand["FR"][1], stand["FR"][2] + 0.08),
            "FL": (stand["FL"][0] + 0.01, stand["FL"][1], stand["FL"][2] + 0.08),
            "RR": (stand["RR"][0] - 0.01, stand["RR"][1], stand["RR"][2] - 0.02),
            "RL": (stand["RL"][0] - 0.01, stand["RL"][1], stand["RL"][2] - 0.02),
        }
        stretch = {
            "FR": (stand["FR"][0] + 0.05, stand["FR"][1], stand["FR"][2] + 0.03),
            "FL": (stand["FL"][0] + 0.05, stand["FL"][1], stand["FL"][2] + 0.03),
            "RR": (stand["RR"][0] - 0.04, stand["RR"][1], stand["RR"][2] + 0.08),
            "RL": (stand["RL"][0] - 0.04, stand["RL"][1], stand["RL"][2] + 0.08),
        }
        sit = {
            "FR": (stand["FR"][0] + 0.02, stand["FR"][1], stand["FR"][2] + 0.06),
            "FL": (stand["FL"][0] + 0.02, stand["FL"][1], stand["FL"][2] + 0.06),
            "RR": (stand["RR"][0] - 0.06, stand["RR"][1], stand["RR"][2] + 0.13),
            "RL": (stand["RL"][0] - 0.06, stand["RL"][1], stand["RL"][2] + 0.13),
        }
        # Front-left: lift foot off the ground (body frame +z). Pure vertical target lets
        # IK split motion across thigh + calf on real Go2; calf-only reads as a twist.
        # Other three feet = nominal stand.
        _fl = stand["FL"]
        three_leg_stand = {
            **stand,
            "FL": (_fl[0], _fl[1], _fl[2] + 0.17),
        }
        return {
            "stand": stand,
            "crouch": crouch,
            "bow": bow,
            "stretch": stretch,
            "sit": sit,
            "three_leg_stand": three_leg_stand,
        }

    def _build_sequence_library(self) -> dict[str, GestureSequence]:
        stand = self._poses["stand"]
        lifted = self._poses["three_leg_stand"]
        _fl = stand["FL"]
        lift_mid = {
            **stand,
            "FL": (_fl[0], _fl[1], _fl[2] + 0.07),
        }
        wave_lift = offset_feet(stand, FL=(0.07, 0.02, 0.14))
        wave_out = offset_feet(wave_lift, FL=(0.0, 0.05, 0.0))
        wave_in = offset_feet(wave_lift, FL=(0.0, -0.05, 0.0))
        # "Shake hand" style: shift weight backward, then lift + reach with one front paw (FR).
        # Real Go2 uses Sport/HighCmd skill IDs over DDS; this is the same idea in foot-target IK.
        weight_prep = offset_feet(
            stand,
            FR=(-0.04, 0.0, -0.015),
            FL=(-0.04, 0.0, -0.015),
            RR=(0.02, 0.0, 0.0),
            RL=(0.02, 0.0, 0.0),
        )
        _frs = weight_prep["FR"]
        shake_extend = {
            **weight_prep,
            "FR": (_frs[0] + 0.11, _frs[1] - 0.05, _frs[2] + 0.16),
        }
        shake_mid = {
            **weight_prep,
            "FR": (_frs[0] + 0.05, _frs[1] - 0.02, _frs[2] + 0.08),
        }
        return {
            "lift_front_left": GestureSequence(
                name="lift_front_left",
                frames=(
                    PoseFrame(duration=0.40, feet=stand),
                    PoseFrame(duration=0.32, feet=lift_mid),
                    PoseFrame(duration=1.10, feet=lifted),
                    PoseFrame(duration=0.28, feet=lifted),
                    PoseFrame(duration=0.28, feet=lift_mid),
                    PoseFrame(duration=0.40, feet=stand),
                ),
                loop=False,
            ),
            "wave_front_left": GestureSequence(
                name="wave_front_left",
                frames=(
                    PoseFrame(duration=0.30, feet=stand),
                    PoseFrame(duration=0.20, feet=wave_lift),
                    PoseFrame(duration=0.20, feet=wave_out),
                    PoseFrame(duration=0.20, feet=wave_in),
                    PoseFrame(duration=0.20, feet=wave_out),
                    PoseFrame(duration=0.25, feet=stand),
                ),
                loop=False,
            ),
            "shake_hand": GestureSequence(
                name="shake_hand",
                frames=(
                    PoseFrame(duration=0.35, feet=stand),
                    PoseFrame(duration=0.38, feet=weight_prep),
                    PoseFrame(duration=0.35, feet=shake_mid),
                    PoseFrame(duration=0.45, feet=shake_extend),
                    PoseFrame(duration=0.55, feet=shake_extend),
                    PoseFrame(duration=0.30, feet=shake_mid),
                    PoseFrame(duration=0.35, feet=weight_prep),
                    PoseFrame(duration=0.40, feet=stand),
                ),
                loop=False,
            ),
        }
