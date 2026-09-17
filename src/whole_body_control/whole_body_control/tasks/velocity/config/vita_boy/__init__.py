import gymnasium as gym

gym.register(
    id="VitaBoy-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "whole_body_control.tasks.velocity.velocity_env_cfg:VelocityRoughEnvCfg",
        "play_env_cfg_entry_point": "whole_body_control.tasks.velocity.velocity_env_cfg:VelocityRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "whole_body_control.tasks.velocity.config.vita_boy.rl_cfg:RoughPPORunnerCfg",
    },
)


gym.register(
    id="VitaBoy-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "whole_body_control.tasks.velocity.velocity_env_cfg:VelocityFlatEnvCfg",
        "play_env_cfg_entry_point": "whole_body_control.tasks.velocity.velocity_env_cfg:VelocityFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "whole_body_control.tasks.velocity.config.vita_boy.rl_cfg:BasePPORunnerCfg",
    },
)
