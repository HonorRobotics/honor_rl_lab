import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from whole_body_control.assets import ISAAC_ASSET_DIR

VITA_BOY_URDF_DIR = ISAAC_ASSET_DIR / "robots" / "vita_boy" / "urdf"

ARMATURE_PG20 = 3.785e-3
STIFFNESS_PG20 = 15.0
DAMPING_PG20 = 1.0
EFFORT_LIMIT_PG20 = 30.0
VELOCITY_LIMIT_PG20 = 37.0

ARMATURE_PG20_PARALLEL = ARMATURE_PG20 * 2.0
STIFFNESS_PG20_PARALLEL = STIFFNESS_PG20 * 2.0
DAMPING_PG20_PARALLEL = DAMPING_PG20 * 2.0
EFFORT_LIMIT_PG20_PARALLEL = EFFORT_LIMIT_PG20 * 2.0
VELOCITY_LIMIT_PG20_PARALLEL = VELOCITY_LIMIT_PG20

ARMATURE_PG30 = 1.334e-2
STIFFNESS_PG30 = 100.0
DAMPING_PG30 = 3.0
EFFORT_LIMIT_PG30 = 88.0
VELOCITY_LIMIT_PG30 = 32.0

ARMATURE_PG40 = 3.340e-2
STIFFNESS_PG40 = 150.0
DAMPING_PG40 = 4.0
EFFORT_LIMIT_PG40 = 139.0
VELOCITY_LIMIT_PG40 = 20.0

ARMATURE_PG50 = 8.973e-2
STIFFNESS_PG50 = 200.0
DAMPING_PG50 = 5.0
EFFORT_LIMIT_PG50 = 150.0
VELOCITY_LIMIT_PG50 = 20.0

ARMATURE_DM3410 = 0.01
STIFFNESS_DM3410 = 10.0
DAMPING_DM3410 = 1.0
EFFORT_LIMIT_DM3410 = 7.0
VELOCITY_LIMIT_DM3410 = 12.5


@configclass
class PG20(ImplicitActuatorCfg):
    stiffness = STIFFNESS_PG20
    damping = DAMPING_PG20
    effort_limit = EFFORT_LIMIT_PG20
    effort_limit_sim = EFFORT_LIMIT_PG20
    velocity_limit = VELOCITY_LIMIT_PG20
    velocity_limit_sim = VELOCITY_LIMIT_PG20
    armature = ARMATURE_PG20


@configclass
class PG20Parallel(ImplicitActuatorCfg):
    stiffness = STIFFNESS_PG20_PARALLEL
    damping = DAMPING_PG20_PARALLEL
    effort_limit = EFFORT_LIMIT_PG20_PARALLEL
    effort_limit_sim = EFFORT_LIMIT_PG20_PARALLEL
    velocity_limit = VELOCITY_LIMIT_PG20_PARALLEL
    velocity_limit_sim = VELOCITY_LIMIT_PG20_PARALLEL
    armature = ARMATURE_PG20_PARALLEL


@configclass
class PG30(ImplicitActuatorCfg):
    stiffness = STIFFNESS_PG30
    damping = DAMPING_PG30
    effort_limit = EFFORT_LIMIT_PG30
    effort_limit_sim = EFFORT_LIMIT_PG30
    velocity_limit = VELOCITY_LIMIT_PG30
    velocity_limit_sim = VELOCITY_LIMIT_PG30
    armature = ARMATURE_PG30


@configclass
class PG40(ImplicitActuatorCfg):
    stiffness = STIFFNESS_PG40
    damping = DAMPING_PG40
    effort_limit = EFFORT_LIMIT_PG40
    effort_limit_sim = EFFORT_LIMIT_PG40
    velocity_limit = VELOCITY_LIMIT_PG40
    velocity_limit_sim = VELOCITY_LIMIT_PG40
    armature = ARMATURE_PG40


@configclass
class PG50(ImplicitActuatorCfg):
    stiffness = STIFFNESS_PG50
    damping = DAMPING_PG50
    effort_limit = EFFORT_LIMIT_PG50
    effort_limit_sim = EFFORT_LIMIT_PG50
    velocity_limit = VELOCITY_LIMIT_PG50
    velocity_limit_sim = VELOCITY_LIMIT_PG50
    armature = ARMATURE_PG50


@configclass
class DM3410(ImplicitActuatorCfg):
    stiffness = STIFFNESS_DM3410
    damping = DAMPING_DM3410
    effort_limit = EFFORT_LIMIT_DM3410
    effort_limit_sim = EFFORT_LIMIT_DM3410
    velocity_limit = VELOCITY_LIMIT_DM3410
    velocity_limit_sim = VELOCITY_LIMIT_DM3410
    armature = ARMATURE_DM3410


VITA_BOY_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{VITA_BOY_URDF_DIR}/main.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "pg20": PG20(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*elbow_joint",
                ".*wrist_roll_joint",
            ]
        ),
        "pg20_parallel": PG20Parallel(
            joint_names_expr=[
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ]
        ),
        "pg30": PG30(joint_names_expr=[".*_hip_yaw_joint"]),
        "pg40": PG40(joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"]),
        "pg50": PG50(joint_names_expr=[".*_hip_pitch_joint", "waist_yaw_joint"]),
        "dm3410": DM3410(joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"]),
    },
)


VITA_BOY_ACTION_SCALE = {}
for actuator in VITA_BOY_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in effort and name in stiffness and stiffness[name]:
            VITA_BOY_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
