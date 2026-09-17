# Honor RL Lab

<div align="center">

**Reinforcement learning environments for locomotion and whole-body motion tracking on the VITA BOY humanoid robot.**

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76B900?logo=nvidia&logoColor=white)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-4B8BBE)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENCE)

[English](README.md) | [简体中文](README_zh.md)

</div>

## ✨ Overview

Honor RL Lab is an [Isaac Lab](https://isaac-sim.github.io/IsaacLab)-based reinforcement learning project for the VITA BOY humanoid robot. It provides an integrated workflow for training and evaluating locomotion and whole-body motion-tracking policies with RSL-RL.

### Features

- **Locomotion** — train velocity-tracking policies on flat or rough terrain.
- **Motion tracking** — train whole-body policies from custom motion data.
- **MuJoCo deployment** — validate exported locomotion and tracking policies in a lightweight simulator.
- **Configurable tasks** — modular observations, rewards, events, commands, and termination conditions.
- **Developer friendly** — editable installation and a clear separation between robot assets, environment configurations, and learning configurations.

## 🧩 Tasks

| Task ID | Description |
| --- | --- |
| `VitaBoy-Velocity-Flat-v0` | Velocity tracking on flat terrain |
| `VitaBoy-Velocity-Rough-v0` | Velocity tracking on rough terrain |
| `VitaBoy-Tracking-v0` | Whole-body reference motion tracking |

## 📦 Setup

### Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3.11
- [Isaac Sim 5.1.0](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
- [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab)

Install Isaac Lab by following its [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). A conda-based installation is recommended for convenient command-line use.

### Install

Clone this repository outside the Isaac Lab installation directory, activate the environment that contains Isaac Lab, and install the project in editable mode:

```bash
conda activate honor_rl_lab
cd honor_rl_lab
python -m pip install -e src/whole_body_control
```

Verify the installation by listing all registered environments:

```bash
python scripts/list_envs.py
```

## 🚀 Usage

### Locomotion

Train a flat-terrain policy:

```bash
python scripts/rsl_rl/train.py \
  --task VitaBoy-Velocity-Flat-v0 \
  --headless \
  --num_envs 4096
```

To train on rough terrain, replace the task ID with `VitaBoy-Velocity-Rough-v0`.

Evaluate a trained policy:

```bash
python scripts/rsl_rl/play.py \
  --task VitaBoy-Velocity-Flat-v0 \
  --checkpoint logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

The evaluation script exports the policy to `exported/policy.onnx` in the same
run directory.

### Motion Tracking

#### Prepare Data

Convert a CSV motion file to the NPZ format used by the tracking environment:

```bash
python datasets/csv_to_npz.py \
  --input-file datasets/motions/vita_boy/dance.csv \
  --output-name datasets/motions/vita_boy/dance.npz \
  --headless
```

#### Train

```bash
python scripts/rsl_rl/train.py \
  --task VitaBoy-Tracking-v0 \
  --motion-file datasets/motions/vita_boy/dance.npz \
  --headless \
  --num_envs 4096
```

#### Evaluate

```bash
python scripts/rsl_rl/play.py \
  --task VitaBoy-Tracking-v0 \
  --motion-file datasets/motions/vita_boy/dance.npz \
  --checkpoint logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

> 💡 **Tip:** `--num_envs 4096` is a starting point for training. Reduce it if GPU memory is limited. Remove `--headless` when a simulator window is needed.

Training outputs are written to:

```text
logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

### MuJoCo Deployment

Place the exported policies at `deploy/checkpoints/loco.onnx` and
`deploy/checkpoints/tracking.onnx`, then run:

```bash
python deploy/run.py
```

Press `R` to enter FIXEDPOSE, then `L` for LOCO or `M` for TRACKING. Hold the
arrow keys and `Q` / `E` to command velocity; press `Esc` to exit. See
[`deploy/README.md`](deploy/README.md) for configuration details.

## 🛠️ Development

The main extension is located in `src/whole_body_control/whole_body_control`. The following files are common starting points for secondary development:

```text
honor_rl_lab/
├── deploy/                         # MuJoCo policy validation
├── datasets/                       # Motion data and tools
├── scripts/                        # Training and evaluation
└── src/whole_body_control/
    └── whole_body_control/
        ├── assets/robots/vita_boy/    # Robot model and configuration
        └── tasks/
            ├── velocity/           # Locomotion
            │   ├── config/vita_boy/   # Task and agent configuration
            │   ├── mdp/            # Task logic
            │   └── velocity_env_cfg.py
            └── tracking/           # Motion tracking
                ├── config/vita_boy/   # Task and agent configuration
                ├── mdp/            # Task logic
                └── tracking_env_cfg.py
```

Typical customization points:

1. Update the robot model and actuator parameters under `assets/robots/vita_boy/`.
2. Adjust observations, rewards, commands, events, or termination conditions under the relevant `tasks/*/mdp/` directory.
3. Tune environment and terrain settings in `*_env_cfg.py` and `terrain_cfg.py`.
4. Tune RSL-RL hyperparameters in `tasks/*/config/vita_boy/rl_cfg.py`.
5. Register a new Gymnasium task in the corresponding `config/vita_boy/__init__.py`, then confirm it with `python scripts/list_envs.py`.

## 📄 License

This project is released under the [Apache License 2.0](LICENCE).

## 🙏 Thanks

Honor RL Lab is inspired by and built upon these excellent open-source projects:

| Project | Contribution |
| :--- | :--- |
| **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** | A modular robot-learning framework built on NVIDIA Isaac Sim. |
| **[LeggedLab](https://github.com/Hellod035/LeggedLab)** | An Isaac Lab-based framework for training legged locomotion policies. |
| **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** | A learning framework for whole-body humanoid motion tracking. |
