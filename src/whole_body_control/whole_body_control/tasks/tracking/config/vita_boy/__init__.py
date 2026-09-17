import gymnasium as gym

gym.register(
    id="VitaBoy-Tracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "whole_body_control.tasks.tracking.tracking_env_cfg:TrackingEnvCfg",
        "play_env_cfg_entry_point": "whole_body_control.tasks.tracking.tracking_env_cfg:TrackingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "whole_body_control.tasks.tracking.config.vita_boy.rl_cfg:BasePPORunnerCfg",
    },
)
