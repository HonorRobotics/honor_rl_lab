# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


def _ppo_algorithm_cfg() -> RslRlPpoAlgorithmCfg:
    """Create the PPO settings shared by flat and rough tasks."""
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Feed-forward PPO configuration for gravel terrain."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50001
    save_interval = 1000
    experiment_name = ""  # same as task name
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _ppo_algorithm_cfg()


@configclass
class RoughPPORunnerCfg(BasePPORunnerCfg):
    """Recurrent PPO configuration for rough terrain."""

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
