"""Observation terms for VITA BOY velocity locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def feet_contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Return one binary contact indicator for each selected foot."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_force = torch.max(torch.linalg.norm(forces, dim=-1), dim=1).values
    return (max_force > threshold).float()
