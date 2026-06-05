# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from pathlib import Path

# ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
ASSET_DIR = str(Path(__file__).resolve().parents[5] / "assets" / "Factory_new")


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05
    center_x_offset = 0.0
    center_y_offset = 0.0
    height_above_center = 0.0

@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_noise: list = [0.0, 0.0, 0.0] 
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_thread task only.
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]  # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]  # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    # Friction between gripper and peg (held asset). If set, overrides held_asset_cfg.friction and robot_cfg.friction.
    gripper_peg_friction: float | None = None

    # Domain randomization: sample gripper-peg friction uniformly in [min, max] at each env reset.
    gripper_peg_friction_randomization: bool = False
    gripper_peg_friction_range: list = [0.5, 0.75]  # [min, max] when randomization is enabled


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019
    center_x_offset = 0.0
    center_y_offset = 0.0
    height_above_center = 0.050

@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    # held_asset_rot_noise: list = [0.0, 0.0, 0.0]      # [0.0, 0.628318, 0.0] for tacsl
    # held_asset_rot_noise: list = [0.0, 0.1, 0.0]      # [0.0, 0.628318, 0.0] for tacsl
    held_asset_rot_noise: list = [0.0, 0.2, 0.0]      # [0.0, 0.628318, 0.0] for tacsl
    # held_asset_rot_noise: list = [0.0, 0.628318, 0.0]      # [0.0, 0.628318, 0.0] for tacsl
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012
    # Gear is 0.025 height, but center at 0.05, and another 0.05 for gear
    # 0.010 gear, 0.015 grab-able
    # height_above_center = 0.020
    # center_x_offset = -0.020
    # center_y_offset = 0.0

@configclass
class GearMesh(FactoryTask):
    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=small_gear_usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=large_gear_usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    # Gears Asset
    add_flanking_gears = True
    add_flanking_gears_prob = 1.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.035]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of gear peg height.
    success_threshold: float = 0.05
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class NutM16(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_nut_m16.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25
    height_above_center = 0.01

@configclass
class BoltM16(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002


@configclass
class NutThread(FactoryTask):
    name = "nut_thread"
    fixed_asset_cfg = BoltM16()
    held_asset_cfg = NutM16()
    asset_size = 16.0
    duration_s = 30.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 1.83]
    hand_init_orn_noise: list = [0.0, 0.0, 0.26]

    # Action
    unidirectional_rot: bool = True

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.003, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_noise: list = [0.0, 0.6, 0.0]  # has visualized to check this is the right direction
    held_asset_rot_init: float = -90.0

    # Reward.
    ee_success_yaw = 0.0
    keypoint_coef_baseline: list = [100, 2]
    keypoint_coef_coarse: list = [500, 2]  # 100, 2
    keypoint_coef_fine: list = [1500, 0]  # 500, 0
    # Fraction of thread-height.
    success_threshold: float = 0.375
    engage_threshold: float = 0.5
    keypoint_scale: float = 0.05

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


TEST_HELD_OBJECT_PRESETS: dict[str, dict] = {
    "peg": {
        "held_asset_cfg_cls": Peg8mm,
        "fixed_init_pos": (0.49, 0.0, 0.0),
        "fixed_init_rot": (1.0, 0.0, 0.0, 0.0),
    },
    "gear": {
        "held_asset_cfg_cls": MediumGear,
        "fixed_init_pos": (0.49, 0.0, 0.0),
        "fixed_init_rot": (1.0, 0.0, 0.0, 0.0),
    },
    "nut": {
        "held_asset_cfg_cls": NutM16,
        "fixed_init_pos": (0.49, 0.0, 0.005),
        # 30 deg about +X so the nut rests on a flat face for grasping.
        "fixed_init_rot": (0.9659, 0.0, 0.0, 0.2588),
    },
}


def _quat_mul_wxyz(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def sync_fixed_peg_spawn_state(task) -> None:
    """Mirror ``fixed_peg_init_pos`` / ``fixed_peg_init_rot`` into ``fixed_peg.init_state``.

    ``TestTask.fixed_peg.init_state`` is captured when the configclass is built. The kinematic
    held object is spawned from ``init_state`` in USD, so overrides to ``fixed_peg_init_rot``
    must be copied here before ``gym.make()`` (see ``TestEnv._reset_fixed_peg`` for runtime).

    Optional fixed yaw about world +Z uses ``fixed_asset_init_orn_deg``.
    """
    pos = tuple(task.fixed_peg_init_pos)
    base_rot = tuple(task.fixed_peg_init_rot)

    yaw_deg = float(task.fixed_asset_init_orn_deg)
    if yaw_deg != 0.0:
        half_yaw = math.radians(yaw_deg) * 0.5
        yaw_quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        rot = _quat_mul_wxyz(yaw_quat, base_rot)
    else:
        rot = base_rot

    task.fixed_peg.init_state.pos = pos
    task.fixed_peg.init_state.rot = rot


def apply_test_held_object_preset(env_cfg) -> None:
    """Set TestTask held-asset USD and fixed spawn pose from ``env_cfg.held_object`` (peg|gear|nut)."""
    key = str(getattr(env_cfg, "held_object", "peg")).lower()
    if key not in TEST_HELD_OBJECT_PRESETS:
        raise ValueError(f"Unknown test held_object '{key}'. Choose from: {list(TEST_HELD_OBJECT_PRESETS)}")

    preset = TEST_HELD_OBJECT_PRESETS[key]
    held_cfg = preset["held_asset_cfg_cls"]()
    task = env_cfg.task
    task.held_asset_cfg = held_cfg
    task.fixed_peg_init_pos = preset["fixed_init_pos"]
    task.fixed_peg_init_rot = preset["fixed_init_rot"]
    task.fixed_peg.spawn.usd_path = held_cfg.usd_path
    task.fixed_peg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=held_cfg.mass)
    sync_fixed_peg_spawn_state(task)


@configclass
class TestTask(FactoryTask):
    """Test environment with Franka robot and a kinematically fixed held object on the table."""
    name = "test"
    duration_s = 5.0

    held_asset_cfg = Peg8mm()
    fixed_asset_cfg = FixedAssetCfg()

    # Fixed peg pose in env frame (upright peg resting on table).
    fixed_peg_init_pos: tuple = (0.49, 0.0, 0.0)
    fixed_peg_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)

    fixed_peg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedPeg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            # Peg USD ships with ArticulationRootAPI; disable for a single kinematic rigid body.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=fixed_peg_init_pos,
            rot=fixed_peg_init_rot,
        ),
    )
    
    # Robot initial end-effector pose (in world coordinates, relative to robot base at origin)
    # If hand_init_pos is [0, 0, 0], default joint positions will be used instead of IK
    # Default: position above table center (0.55, 0.0, 0.3) with gripper pointing down
    hand_init_pos: list = [0.5, 0.0, 0.06]  # World coordinates (x, y, z) in meters
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]  # Optional noise for randomization
    hand_init_orn: list = [3.1416, 0.0, 0.0]  # Euler angles (roll, pitch, yaw) in radians
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]  # Optional noise for randomization
    
    # No randomization needed
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0
