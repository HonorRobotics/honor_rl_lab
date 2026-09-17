"""Convert a VITA BOY motion CSV file to the NPZ format used by the tracking task.

The input CSV is expected to contain one frame per row with the following layout:
root position (3), root quaternion in xyzw order (4), and VITA BOY joint positions.
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a VITA BOY motion CSV file to an Isaac Lab tracking NPZ file.")
parser.add_argument("--input-file", "-f", type=str, required=True, help="Path to the input motion CSV file.")
parser.add_argument("--input-fps", type=int, default=50, help="FPS of the input motion.")
parser.add_argument("--output-fps", type=int, default=50, help="FPS of the output motion.")
parser.add_argument("--output-name", type=str, help="Output NPZ path. Defaults to the input path with .npz suffix.")
parser.add_argument(
    "--frame-range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="1-indexed inclusive frame range. If omitted, all frames are used.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.output_name is None:
    args_cli.output_name = args_cli.input_file.rsplit(".", 1)[0] + ".npz"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_control.assets.robots.vita_boy import VITA_BOY_CFG as ROBOT_CFG

VITA_BOY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for replaying a VITA BOY motion."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        if motion.shape[1] != 7 + len(VITA_BOY_JOINT_NAMES):
            raise ValueError(
                f"Expected {7 + len(VITA_BOY_JOINT_NAMES)} columns, got {motion.shape[1]} in {self.motion_file}."
            )
        self.motion_base_pos_input = motion[:, :3]
        self.motion_base_rot_input = motion[:, 3:7]
        self.motion_base_rot_input = self.motion_base_rot_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_joint_pos_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_pos = self._lerp(
            self.motion_base_pos_input[index_0], self.motion_base_pos_input[index_1], blend.unsqueeze(1)
        )
        self.motion_base_rot = self._slerp(
            self.motion_base_rot_input[index_0], self.motion_base_rot_input[index_1], blend
        )
        self.motion_joint_pos = self._lerp(
            self.motion_joint_pos_input[index_0], self.motion_joint_pos_input[index_1], blend.unsqueeze(1)
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vel = torch.gradient(self.motion_base_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_joint_vel = torch.gradient(self.motion_joint_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vel = self._so3_derivative(self.motion_base_rot, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_pos[self.current_idx : self.current_idx + 1],
            self.motion_base_rot[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vel[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vel[self.current_idx : self.current_idx + 1],
            self.motion_joint_pos[self.current_idx : self.current_idx + 1],
            self.motion_joint_vel[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(VITA_BOY_JOINT_NAMES, preserve_order=True)[0]

    log = {
        "fps": np.array([args_cli.output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_joint_pos,
            motion_joint_vel,
        ), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_joint_pos
        joint_vel[:, robot_joint_indexes] = motion_joint_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[key] = np.stack(log[key], axis=0)
            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
