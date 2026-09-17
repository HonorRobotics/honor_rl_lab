# Honor RL Lab

<div align="center">

**面向元气仔（VITA BOY）人形机器人的运动控制与全身动作跟踪强化学习环境。**

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76B900?logo=nvidia&logoColor=white)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-4B8BBE)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENCE)

[English](README.md) | [简体中文](README_zh.md)

</div>

## ✨ 简介

Honor RL Lab 是一个面向元气仔人形机器人的强化学习项目，基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab) 构建，提供使用 RSL-RL 训练和评估运动控制与全身动作跟踪策略的一体化工作流。

### 特性

- **运动控制** — 在平地或粗糙地形上训练速度跟踪策略。
- **动作跟踪** — 使用自定义动作数据训练全身动作跟踪策略。
- **MuJoCo 部署** — 在轻量仿真器中验证导出的行走与动作跟踪策略。
- **灵活配置** — 以模块化方式配置观测、奖励、事件、命令和终止条件。
- **便于二次开发** — 支持可编辑安装，并清晰分离机器人资产、环境配置与训练配置。

## 🧩 任务

| 任务 ID | 说明 |
| --- | --- |
| `VitaBoy-Velocity-Flat-v0` | 平地速度跟踪 |
| `VitaBoy-Velocity-Rough-v0` | 粗糙地形速度跟踪 |
| `VitaBoy-Tracking-v0` | 全身参考动作跟踪 |

## 📦 安装

### 要求

- Linux（已在 Ubuntu 22.04 上测试通过）
- Python 3.11
- [Isaac Sim 5.1.0](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
- [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab)

请先按照 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 完成安装。推荐使用 conda 安装方式，以便在终端中直接调用 Python 脚本。

### 安装项目

将本仓库克隆到 Isaac Lab 安装目录之外，激活已安装 Isaac Lab 的环境，然后以可编辑模式安装本项目：

```bash
conda activate honor_rl_lab
cd honor_rl_lab
python -m pip install -e src/whole_body_control
```

列出已注册的环境，验证安装是否成功：

```bash
python scripts/list_envs.py
```

## 🚀 使用

### 运动控制

训练平地运动策略：

```bash
python scripts/rsl_rl/train.py \
  --task VitaBoy-Velocity-Flat-v0 \
  --headless \
  --num_envs 4096
```

如需训练粗糙地形策略，请将任务 ID 替换为 `VitaBoy-Velocity-Rough-v0`。

评估已训练的策略：

```bash
python scripts/rsl_rl/play.py \
  --task VitaBoy-Velocity-Flat-v0 \
  --checkpoint logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

评估脚本会同时将策略导出到当前运行目录下的 `exported/policy.onnx`。

### 动作跟踪

#### 准备数据

将 CSV 动作文件转换为跟踪环境使用的 NPZ 格式：

```bash
python datasets/csv_to_npz.py \
  --input-file datasets/motions/vita_boy/dance.csv \
  --output-name datasets/motions/vita_boy/dance.npz \
  --headless
```

#### 训练

```bash
python scripts/rsl_rl/train.py \
  --task VitaBoy-Tracking-v0 \
  --motion-file datasets/motions/vita_boy/dance.npz \
  --headless \
  --num_envs 4096
```

#### 评估

```bash
python scripts/rsl_rl/play.py \
  --task VitaBoy-Tracking-v0 \
  --motion-file datasets/motions/vita_boy/dance.npz \
  --checkpoint logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

> 💡 **提示：** `--num_envs 4096` 是建议的训练起点；如果 GPU 显存不足，可适当减小该值。需要显示仿真窗口时，请移除 `--headless`。

训练结果默认保存在：

```text
logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

### MuJoCo 部署

将导出的策略分别放置到 `deploy/checkpoints/loco.onnx` 和
`deploy/checkpoints/tracking.onnx`，然后运行：

```bash
python deploy/run.py
```

按 `R` 进入 FIXEDPOSE，再按 `L` 进入 LOCO 或按 `M` 进入 TRACKING。按住方向键和
`Q` / `E` 输入速度指令，按 `Esc` 退出。配置方法见
[`deploy/README.md`](deploy/README.md)。

## 🛠️ 二次开发

项目的主要扩展代码位于 `src/whole_body_control/whole_body_control`。二次开发时可重点关注以下目录与文件：

```text
honor_rl_lab/
├── deploy/                         # MuJoCo 策略验证
├── datasets/                       # 动作数据与工具
├── scripts/                        # 训练与评估入口
└── src/whole_body_control/
    └── whole_body_control/
        ├── assets/robots/vita_boy/    # 机器人模型与配置
        └── tasks/
            ├── velocity/           # 运动控制
            │   ├── config/vita_boy/   # 任务与智能体配置
            │   ├── mdp/            # 任务逻辑
            │   └── velocity_env_cfg.py
            └── tracking/           # 动作跟踪
                ├── config/vita_boy/   # 任务与智能体配置
                ├── mdp/            # 任务逻辑
                └── tracking_env_cfg.py
```

常见的定制入口：

1. 在 `assets/robots/vita_boy/` 中更新机器人模型与执行器参数。
2. 在对应的 `tasks/*/mdp/` 目录中调整观测、奖励、命令、事件或终止条件。
3. 在 `*_env_cfg.py` 和 `terrain_cfg.py` 中调整环境与地形参数。
4. 在 `tasks/*/config/vita_boy/rl_cfg.py` 中调整 RSL-RL 超参数。
5. 在对应的 `config/vita_boy/__init__.py` 中注册新的 Gymnasium 任务，并运行 `python scripts/list_envs.py` 确认注册结果。

## 📄 许可

本项目采用 [Apache License 2.0](LICENCE) 开源。

## 🙏 致谢

Honor RL Lab 的实现离不开以下优秀开源项目的启发与支持：

| 项目 | 贡献 |
| :--- | :--- |
| **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** | 基于 NVIDIA Isaac Sim 构建的模块化机器人学习框架。 |
| **[LeggedLab](https://github.com/Hellod035/LeggedLab)** | 基于 Isaac Lab 的足式机器人运动策略训练框架。 |
| **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** | 面向人形机器人全身动作跟踪的学习框架。 |
