# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from omegaconf import OmegaConf
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import (
    ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask,
    ForgePegInsertTac, ForgeGearMeshTac, ForgeNutThreadTac
)

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    # ema_factor_range = [0.2, 0.2]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    # default_task_prop_gains = [300.0, 300.0, 300.0, 28.0, 28.0, 28.0]
    # default_task_prop_gains = [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
    # default_task_prop_gains = [200.0, 200.0, 200.0, 30.0, 30.0, 30.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]

    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    action_rep: str = "rel_ee_pose"
    joint_pos_threshold = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # For action_rep="delta_ee_pose": either re-anchor current_pose + delta on
    # every decimation substep (legacy), or snapshot one absolute target at the
    # policy/control-step boundary and hold it throughout decimation.
    delta_target_mode: str = "per_physics_step"
    # Legacy Forge averages the two finger Jacobians. RobustTrack controls the
    # actual tool body's origin and shifts its COM-referenced Jacobian/velocity.
    tool_kinematics_mode: str = "finger_average"

    # Use the full operational-space control law: premultiply the task-space PD
    # wrench by the task inertia Λ = (J M⁻¹ Jᵀ)⁻¹. False falls back to the legacy
    # raw Jᵀ mapping while still using the shared factory controller.
    use_task_space_inertia: bool = True

    # Mass-matrix source for the OSC task-space inertia Λ = (J M⁻¹ Jᵀ)⁻¹.
    # True (default): use PhysX's ground-truth generalized mass matrix, which
    # reflects any payload/link-mass/armature randomization (perfectly-modeled
    # controller). False: rebuild the 7x7 arm mass matrix from the nominal
    # (config-default) inertial parameters via per-link jacobians, keeping the
    # controller blind to the sim-side dynamics randomization. Set False to match
    # the franka_robust_track tracker's default (use_gt_mass_matrix=False) so a
    # policy trained there maps faithfully onto this env.
    use_gt_mass_matrix: bool = True
    


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0
    # fixed asset position noise std: 0.001


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            # "static_friction_range": (0.3, 0.3),
            # "dynamic_friction_range": (0.3, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            # "static_friction_range": (0.25, 0.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            # "static_friction_range": (0.9, 0.9),
            # "dynamic_friction_range": (0.9, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25
    use_dead_zone: bool = True
    policy_use_held_state: bool = False
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        # "held_pos", "held_quat"
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]

    def update_env_params(self):
        super().update_env_params()
        env = self.params.env

        if env.get("policy_use_held_state", None) is not None:
            self.policy_use_held_state = env.policy_use_held_state
        if self.policy_use_held_state:
            self.obs_order = self.obs_order + ["held_pos", "held_quat"]

        obs_rand = env.get("obs_rand", OmegaConf.create({}))
        if obs_rand.get("fingertip_pos", None) is not None:
            self.obs_rand.fingertip_pos = obs_rand.fingertip_pos

        if env.get("use_dead_zone", None) is not None:
            self.use_dead_zone = env.use_dead_zone
        ctrl = env.get("ctrl", OmegaConf.create({}))
        if ctrl.get("ema_factor_range", None) is not None:
            self.ctrl.ema_factor_range = OmegaConf.to_container(ctrl.ema_factor_range, resolve=True)

        if ctrl.get("action_rep", None) is not None:
            self.ctrl.action_rep = str(ctrl.action_rep)

        if ctrl.get("delta_target_mode", None) is not None:
            self.ctrl.delta_target_mode = str(ctrl.delta_target_mode)

        if ctrl.get("use_task_space_inertia", None) is not None:
            self.ctrl.use_task_space_inertia = ctrl.use_task_space_inertia

        if ctrl.get("use_gt_mass_matrix", None) is not None:
            self.ctrl.use_gt_mass_matrix = ctrl.use_gt_mass_matrix

        if ctrl.get("randomize_controller", None) is not None:
            self.ctrl.randomize_controller = ctrl.randomize_controller
            if not self.ctrl.randomize_controller:
                self.ctrl.task_prop_gains_noise_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.ctrl.pos_threshold_noise_level = [0.0, 0.0, 0.0]
                self.ctrl.rot_threshold_noise_level = [0.0, 0.0, 0.0]

        if ctrl.get("default_task_prop_gains", None) is not None:
            self.ctrl.default_task_prop_gains = OmegaConf.to_container(ctrl.default_task_prop_gains, resolve=True)

        if ctrl.get("pos_action_threshold", None) is not None:
            self.ctrl.pos_action_threshold = OmegaConf.to_container(ctrl.pos_action_threshold, resolve=True)

        if ctrl.get("rot_action_threshold", None) is not None:
            self.ctrl.rot_action_threshold = OmegaConf.to_container(ctrl.rot_action_threshold, resolve=True)
        task = env.get("task", OmegaConf.create({}))
        if task.get("hole_xy_scale", None) is not None:
            self.task.hole_xy_scale = float(task.hole_xy_scale)
        if "peg_insert" in self.task.name:
            hole_xy_scale = float(self.task.hole_xy_scale)
            if not math.isfinite(hole_xy_scale) or hole_xy_scale <= 0.0:
                raise ValueError(
                    f"env.task.hole_xy_scale must be finite and positive, got {hole_xy_scale}."
                )
            self.task.fixed_asset.spawn.scale = (hole_xy_scale, hole_xy_scale, 1.0)
        if task.get("contact_penalty_threshold_range", None) is not None:
            self.task.contact_penalty_threshold_range = OmegaConf.to_container(task.contact_penalty_threshold_range, resolve=True)
        if task.get("action_penalty_asset_scale", None) is not None:
            self.task.action_penalty_asset_scale = task.action_penalty_asset_scale
        if task.get("action_grad_penalty_scale", None) is not None:
            self.task.action_grad_penalty_scale = task.action_grad_penalty_scale
        if task.get("ee_speed_penalty_scale", None) is not None:
            self.task.ee_speed_penalty_scale = task.ee_speed_penalty_scale
        if task.get("ee_speed_penalty_threshold", None) is not None:
            self.task.ee_speed_penalty_threshold = task.ee_speed_penalty_threshold
        if task.get("ee_speed_exp_penalty_scale", None) is not None:
            self.task.ee_speed_exp_penalty_scale = task.ee_speed_exp_penalty_scale
        if task.get("ee_speed_exp_penalty_k", None) is not None:
            self.task.ee_speed_exp_penalty_k = task.ee_speed_exp_penalty_k
        if task.get("ee_speed_exp_penalty_cap", None) is not None:
            self.task.ee_speed_exp_penalty_cap = task.ee_speed_exp_penalty_cap
        if task.get("ee_ang_speed_penalty_scale", None) is not None:
            self.task.ee_ang_speed_penalty_scale = task.ee_ang_speed_penalty_scale
        if task.get("ee_ang_speed_penalty_threshold", None) is not None:
            self.task.ee_ang_speed_penalty_threshold = task.ee_ang_speed_penalty_threshold
        if task.get("ee_upright_penalty_scale", None) is not None:
            self.task.ee_upright_penalty_scale = task.ee_upright_penalty_scale
        if task.get("contact_penalty_scale", None) is not None:
            self.task.contact_penalty_scale = task.contact_penalty_scale
        if task.get("contact_penalty_cap", None) is not None:
            self.task.contact_penalty_cap = task.contact_penalty_cap
        # >1.0 => batch success can never reach it, so the success-prediction penalty
        # never latches on (i.e. success_pred_scale stays 0 / reward term disabled).
        if task.get("delay_until_ratio", None) is not None:
            self.task.delay_until_ratio = task.delay_until_ratio
    

    def __post_init__(self):
        super().__post_init__()

        

@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0


# =============================================================================
# Tactile Finger Forge Environment Configurations
# =============================================================================

@configclass
class ForgeTaskPegInsertTacCfg(ForgeEnvCfg):
    """Forge peg insertion environment for tactile fingers."""
    task_name = "peg_insert_tac"
    task = ForgePegInsertTac()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshTacCfg(ForgeEnvCfg):
    """Forge gear mesh environment for tactile fingers."""
    task_name = "gear_mesh_tac"
    task = ForgeGearMeshTac()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadTacCfg(ForgeEnvCfg):
    """Forge nut threading environment for tactile fingers."""
    task_name = "nut_thread_tac"
    task = ForgeNutThreadTac()
    episode_length_s = 30.0
