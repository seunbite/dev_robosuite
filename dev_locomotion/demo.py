from __future__ import annotations

import argparse

from .commands import VelocityCommand
from .controllers.high_level import HighLevelController
from .mujoco_env import DependencyError, MujocoQuadrupedEnv


def build_policy(controller: HighLevelController, behavior: str):
    if behavior in controller.gestures.pose_names:
        return lambda t, obs: controller.command_pose(behavior)

    if behavior in controller.gestures.sequence_names:
        return lambda t, obs: controller.command_gesture(behavior, t)

    if behavior == "walk_forward":
        command = VelocityCommand(vx=0.35)
        return lambda t, obs: controller.command_walk(command, t)

    if behavior == "walk_side":
        command = VelocityCommand(vx=0.0, vy=0.15)
        return lambda t, obs: controller.command_walk(command, t)

    if behavior == "turn_left":
        command = VelocityCommand(vx=0.0, vy=0.0, wz=0.9)
        return lambda t, obs: controller.command_walk(command, t)

    raise ValueError(f"Unknown behavior: {behavior}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quadruped locomotion / gesture demos.")
    parser.add_argument("--mode", choices=("viewer", "headless"), default="viewer")
    parser.add_argument(
        "--behavior",
        default="walk_forward",
        help="stand, crouch, bow, stretch, sit, three_leg_stand, wave_front_left, shake_hand, lift_front_left, walk_forward, walk_side, turn_left",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--xml-path", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = HighLevelController()

    try:
        env = MujocoQuadrupedEnv(xml_path=args.xml_path)
    except DependencyError as exc:
        print(exc)
        return 1

    policy = build_policy(controller, args.behavior)

    if args.mode == "viewer":
        env.launch_passive_viewer(policy, duration_s=args.duration, controller=controller.low_level)
        return 0

    steps = int(args.duration / controller.low_level.kinematics.config.control_dt)
    observation = env.reset()
    for _ in range(steps):
        command = policy(float(observation["time"]), observation)
        observation = env.step_joint_command(command, controller=controller.low_level)

    print(
        "final base position:",
        tuple(round(value, 4) for value in observation["base_position"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
