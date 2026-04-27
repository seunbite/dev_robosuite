from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .commands import JointMap, LowLevelCommand
from .config import JOINT_ORDER
from .controllers.low_level import LowLevelController
from .kinematics import QuadrupedKinematics


class DependencyError(RuntimeError):
    pass


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise DependencyError(
            "mujoco 패키지가 필요합니다. `python3 -m pip install mujoco` 후 다시 실행해 주세요."
        ) from exc
    return mujoco


@dataclass
class MujocoQuadrupedEnv:
    xml_path: str | Path | None = None
    substeps: int = 10
    kinematics: QuadrupedKinematics = field(default_factory=QuadrupedKinematics)
    low_level: LowLevelController = field(init=False)
    #: Absolute path passed to MuJoCo (after resolving ``go2`` / ``simple`` presets).
    asset_xml_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self._mujoco = _import_mujoco()
        self.low_level = LowLevelController(self.kinematics)
        asset_path = self._resolve_xml_path(self.xml_path)
        self.asset_xml_path = asset_path.resolve()
        self.model = self._mujoco.MjModel.from_xml_path(str(self.asset_xml_path))
        self.data = self._mujoco.MjData(self.model)
        self._warn_if_go2_pitch_axes_wrong()
        self._qpos_index: dict[str, int] = {}
        self._qvel_index: dict[str, int] = {}
        self._ctrl_index: dict[str, int] = {}
        self._freejoint_qposadr: int | None = None
        self._freejoint_dofadr: int | None = None
        self._build_name_maps()
        self.reset()

    @staticmethod
    def default_xml_path() -> Path:
        return Path(__file__).resolve().parent / "assets" / "quadruped.xml"

    @staticmethod
    def go2_xml_path() -> Path:
        candidate = Path(__file__).resolve().parents[2] / "dev_robosuite" / "robosuite" / "models" / "assets" / "robots" / "go2" / "robot.xml"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find local Go2 asset: {candidate}")
        return candidate

    @classmethod
    def _resolve_xml_path(cls, xml_path: str | Path | None) -> Path:
        if xml_path is None or str(xml_path) == "simple":
            return cls.default_xml_path()
        if str(xml_path) == "go2":
            return cls.go2_xml_path()
        return Path(xml_path)

    def _warn_if_go2_pitch_axes_wrong(self) -> None:
        """If thigh/knee default to world Z, the leg link lies on the joint axis and the foot barely moves."""
        mujoco = self._mujoco
        path_s = str(self.asset_xml_path).replace("\\", "/")
        if "/robots/go2/" not in path_s and not path_s.endswith("/go2/robot.xml"):
            return
        thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FL_thigh_joint")
        calf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FL_calf_joint")
        if thigh_id < 0 or calf_id < 0:
            return
        thigh_axis = self.model.jnt_axis[thigh_id]
        calf_axis = self.model.jnt_axis[calf_id]
        # Expect pitch ~ body Y (0,1,0) in this MJCF; wrong presets give (0,0,1).
        if abs(thigh_axis[1]) < 0.9 or abs(calf_axis[1]) < 0.9:
            warnings.warn(
                f"Go2 FL thigh/calf joint axes look like {thigh_axis} / {calf_axis} "
                f"(expected pitch ~ (0,1,0)); foot IK will not fold the leg. "
                f"Loaded XML: {self.asset_xml_path}",
                stacklevel=2,
            )

    def _build_name_maps(self) -> None:
        mujoco = self._mujoco
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self._freejoint_qposadr = int(self.model.jnt_qposadr[joint_id])
                self._freejoint_dofadr = int(self.model.jnt_dofadr[joint_id])
                break

        for joint_name in JOINT_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self._qpos_index[joint_name] = int(self.model.jnt_qposadr[joint_id])
            self._qvel_index[joint_name] = int(self.model.jnt_dofadr[joint_id])

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name in self._qpos_index:
                self._ctrl_index[joint_name] = actuator_id

        for joint_name in JOINT_ORDER:
            if joint_name in self._ctrl_index:
                continue
            actuator_name = joint_name.replace("_joint", "_motor")
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                self._ctrl_index[joint_name] = int(actuator_id)

    def reset(self) -> dict[str, object]:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_default_base_pose()

        for joint_name, value in self.kinematics.config.default_joint_map().items():
            self.data.qpos[self._qpos_index[joint_name]] = value

        self._mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def _set_default_base_pose(self) -> None:
        if self._freejoint_qposadr is None:
            return
        qposadr = self._freejoint_qposadr
        self.data.qpos[qposadr : qposadr + 3] = (0.0, 0.0, self.kinematics.config.nominal_base_height + 0.02)
        self.data.qpos[qposadr + 3 : qposadr + 7] = (1.0, 0.0, 0.0, 0.0)

    def hold_base_pose(
        self,
        position: tuple[float, float, float] | None = None,
        quat_wxyz: tuple[float, float, float, float] | None = None,
    ) -> None:
        if self._freejoint_qposadr is None or self._freejoint_dofadr is None:
            return
        qposadr = self._freejoint_qposadr
        dofadr = self._freejoint_dofadr
        position = position or (0.0, 0.0, self.kinematics.config.nominal_base_height + 0.02)
        quat_wxyz = quat_wxyz or (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[qposadr : qposadr + 3] = position
        self.data.qpos[qposadr + 3 : qposadr + 7] = quat_wxyz
        self.data.qvel[dofadr : dofadr + 6] = 0.0

    def apply_joint_positions(
        self,
        joint_positions: Mapping[str, float],
        hold_base: bool = False,
        position: tuple[float, float, float] | None = None,
        quat_wxyz: tuple[float, float, float, float] | None = None,
    ) -> dict[str, object]:
        for joint_name, value in joint_positions.items():
            if joint_name not in self._qpos_index:
                continue
            self.data.qpos[self._qpos_index[joint_name]] = float(value)
            self.data.qvel[self._qvel_index[joint_name]] = 0.0
        if hold_base:
            self.hold_base_pose(position=position, quat_wxyz=quat_wxyz)
        self._mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def joint_positions(self) -> JointMap:
        return {joint: float(self.data.qpos[index]) for joint, index in self._qpos_index.items()}

    def joint_velocities(self) -> JointMap:
        return {joint: float(self.data.qvel[index]) for joint, index in self._qvel_index.items()}

    def observation(self) -> dict[str, object]:
        return {
            "time": float(self.data.time),
            "base_position": tuple(float(v) for v in self.data.qpos[0:3]),
            "base_orientation_wxyz": tuple(float(v) for v in self.data.qpos[3:7]),
            "base_linear_angular_velocity": tuple(float(v) for v in self.data.qvel[0:6]),
            "joint_position": self.joint_positions(),
            "joint_velocity": self.joint_velocities(),
        }

    def step_torques(self, torques: Mapping[str, float], steps: int | None = None) -> dict[str, object]:
        substeps = steps if steps is not None else self.substeps
        self.data.ctrl[:] = 0.0
        for joint_name, value in torques.items():
            if joint_name not in self._ctrl_index:
                continue
            self.data.ctrl[self._ctrl_index[joint_name]] = float(value)
        for _ in range(substeps):
            self._mujoco.mj_step(self.model, self.data)
        return self.observation()

    def step_joint_command(
        self,
        command: LowLevelCommand,
        controller: LowLevelController | None = None,
        steps: int | None = None,
    ) -> dict[str, object]:
        controller = controller or self.low_level
        torques = controller.compute_torques(command, self.joint_positions(), self.joint_velocities())
        return self.step_torques(torques, steps=steps)

    def launch_passive_viewer(
        self,
        policy: Callable[[float, Mapping[str, object]], LowLevelCommand],
        duration_s: float,
        controller: LowLevelController | None = None,
        realtime: bool = True,
    ) -> None:
        controller = controller or self.low_level
        try:
            import mujoco.viewer as viewer
        except ImportError as exc:
            raise DependencyError("mujoco.viewer 를 사용할 수 없습니다. GUI 환경인지 확인해 주세요.") from exc

        step_dt = self.model.opt.timestep * self.substeps
        with viewer.launch_passive(self.model, self.data) as handle:
            start = time.perf_counter()
            while handle.is_running() and self.data.time < duration_s:
                loop_start = time.perf_counter()
                observation = self.observation()
                command = policy(float(self.data.time), observation)
                self.step_joint_command(command, controller=controller)
                handle.sync()
                if realtime:
                    elapsed = time.perf_counter() - loop_start
                    sleep_for = max(0.0, step_dt - elapsed)
                    if sleep_for > 0.0:
                        time.sleep(sleep_for)
            if realtime:
                total_elapsed = time.perf_counter() - start
                _ = total_elapsed
