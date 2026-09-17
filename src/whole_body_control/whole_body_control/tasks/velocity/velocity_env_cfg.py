"""Manager-based locomotion tasks for the VITA BOY humanoid."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_control.assets.robots.vita_boy import VITA_BOY_CFG as ROBOT_CFG
from whole_body_control.tasks.velocity import mdp
from whole_body_control.tasks.velocity.terrain_cfg import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG

OBSERVATION_CLIP = (-100.0, 100.0)
FEET_PATTERN = ".*ankle_roll.*"


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Terrain, VITA BOY robot, and contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization and reset events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """Heading-aware velocity commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """VITA BOY joint-position actions retained from the original task."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Actor and asymmetric critic observations without height scans."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=OBSERVATION_CLIP,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=OBSERVATION_CLIP,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=OBSERVATION_CLIP,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=OBSERVATION_CLIP,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=OBSERVATION_CLIP,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=OBSERVATION_CLIP)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=OBSERVATION_CLIP)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=OBSERVATION_CLIP)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=OBSERVATION_CLIP,
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=OBSERVATION_CLIP)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, clip=OBSERVATION_CLIP)
        last_action = ObsTerm(func=mdp.last_action, clip=OBSERVATION_CLIP)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=OBSERVATION_CLIP)
        feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN)},
            clip=OBSERVATION_CLIP,
        )

        def __post_init__(self):
            self.history_length = 1
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FlatRewardsCfg:
    """Velocity locomotion rewards mapped to VITA BOY body and joint names."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN),
            "threshold": 0.4,
            "command_name": "base_velocity",
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_PATTERN),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.biped_feet_too_near,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_PATTERN),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_PATTERN)},
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"],
            ),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"],
            ),
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"],
            ),
        },
    )


@configclass
class RoughRewardsCfg(FlatRewardsCfg):
    """Rough-terrain reward weights."""

    def __post_init__(self):
        self.feet_air_time.weight = 0.25
        self.track_lin_vel_xy_exp.weight = 1.5
        self.track_ang_vel_z_exp.weight = 1.5
        self.lin_vel_z_l2.weight = -0.25


@configclass
class TerminationsCfg:
    """Terminate on timeout or VITA BOY torso contact."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Manager-based rough-terrain progression."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class VelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Rough-terrain velocity task using the VITA BOY robot."""

    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class VelocityRoughPlayEnvCfg(VelocityRoughEnvCfg):
    """Reduced-size rough-terrain environment for evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10


@configclass
class VelocityFlatEnvCfg(VelocityRoughEnvCfg):
    """Gravel velocity task using the VITA BOY robot."""

    rewards: FlatRewardsCfg = FlatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GRAVEL_TERRAINS_CFG.replace()
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.history_length = 10
        self.observations.critic.history_length = 10
        self.curriculum.terrain_levels = None


@configclass
class VelocityFlatPlayEnvCfg(VelocityFlatEnvCfg):
    """Reduced-size gravel environment for evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
