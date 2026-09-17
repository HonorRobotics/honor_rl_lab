"""Reward terms for VITA BOY velocity locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar velocity tracking in the heading-only frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking about the world z-axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the norm of absolute mechanical joint power."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.linalg.norm(power, dim=-1)


def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize phases in which neither foot contacts the ground."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    is_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1).values > threshold
    return (torch.sum(is_contact, dim=-1) < 0.5).float()


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward sustained single-foot support while a motion command is active."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1).values
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > 0.1
    return reward * moving


def body_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
    max_reward: float = 400.0,
) -> torch.Tensor:
    """Penalize excessive combined vertical force on selected bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2], dim=-1)
    return torch.where(force < threshold, 0.0, force - threshold).clamp(max=max_reward)


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize roll and pitch tilt of a selected body."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    projected_gravity = math_utils.quat_apply_inverse(body_quat, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet whose horizontal contact force dominates vertical force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.linalg.norm(force[..., :2], dim=-1) > 5.0 * torch.abs(force[..., 2]), dim=1)


def biped_feet_too_near(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize three-dimensional foot separation below a threshold."""
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("biped_feet_too_near requires exactly two selected bodies")
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    distance = torch.linalg.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0.0)
