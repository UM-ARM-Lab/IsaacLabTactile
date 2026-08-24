# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import torch

from force_tool.utils.action_penalties import (
    action_alternation_penalty,
    action_curvature_huber_penalty,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import (
    CUBOID_MARKER_CFG,
    FRAME_MARKER_CFG,
    SPHERE_MARKER_CFG,
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils

from isaaclab_tasks.direct.factory import factory_control, factory_utils

from .actuator_shaping import shape_torque_held_command
from .franka_robust_track_env_cfg import FrankaRobustTrackEnvCfg
from .inertial_models import map_urdf_inertials
from .motion_regularization import (
    ee_acceleration_and_jerk_norms,
    raw_angular_acceleration_and_jerk_norms,
    raw_linear_acceleration_and_jerk_norms,
    residual_huber_motion_costs,
)
from .reference_clock import (
    adaptive_bin_sampling_probabilities,
    adaptive_phase_sampling_probabilities,
    axis_aligned_containment_mask,
    dataset_reference_end_timeouts,
    history_seed_offsets,
    is_reference_boundary,
    phase_bin_indices,
    policy_step_to_reference_index,
    reference_coordinates,
    sample_adaptive_phase_start_times,
    sample_variable_reference_starts,
    valid_step_weighted_start_times,
    variable_reference_bin_indices,
    variable_reference_bin_layout,
    virtual_contact_reference_coordinates,
)
from .virtual_contact import (
    contact_coupled_torque,
    compute_virtual_plane_contact,
    construct_reference_plane_trajectory,
    construct_rotational_equilibrium_trajectory,
    descale_virtual_sensor_wrench,
    finite_difference_quaternion_angular_velocity,
    finite_difference_surface_velocity,
    invert_ema_sequence,
    orientation_power_law_contact_torque,
    orientation_spring_contact_torque,
    perturb_reference_plane_trajectory,
    power_law_coefficient_from_reference,
    quaternion_nlerp,
    sensor_reaction_to_world_external,
    smooth_force_directions,
    world_external_to_sensor_reaction,
    world_force_components,
    x_axis_to_vector_quat,
    z_axis_to_vector_quat,
)


_AUTO_TOOL_BODY_CANDIDATES = ("panda_fingertip_centered", "force_sensor", "panda_hand")
_IDENTITY_TOOL_OFFSET = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

# Virtual frames are (parent body, parent->tool position, parent->tool quaternion).
# From panda_arm_wsg_horizontal.urdf, hand->fr3_wsg_tcp is z=0.271 m,
# yaw=+45 degrees. These frames are inserted into cuRobo's tree as fixed links.
_VIRTUAL_TOOL_FRAMES = {
    "fr3_wsg_tcp": (
        "panda_hand",
        (0.0, 0.0, 0.271),
        (0.9238795325, 0.0, 0.0, 0.3826834324),
    ),
}


class FrankaRobustTrackEnv(DirectRLEnv):
    """Robot-only Franka direct env for robust Cartesian command tracking."""

    cfg: FrankaRobustTrackEnvCfg

    def __init__(self, cfg: FrankaRobustTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)
        self.current_action_rep = str(getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose"))
        self.delta_target_mode = str(getattr(self.cfg.ctrl, "delta_target_mode", "per_physics_step"))
        if self.delta_target_mode not in ("per_physics_step", "per_control_step"):
            raise ValueError(
                "ctrl.delta_target_mode must be 'per_physics_step' or 'per_control_step', "
                f"got {self.delta_target_mode!r}."
            )
        self._action_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._action_target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self._previous_action_target_pos = self._action_target_pos.clone()
        self._previous_action_target_quat = self._action_target_quat.clone()
        delay_range = getattr(self.cfg.ctrl, "delay_range", None)
        if delay_range is None:
            fixed_delay = int(self.cfg.ctrl.target_update_delay_substeps)
            delay_range = [fixed_delay, fixed_delay]
        else:
            delay_range = [int(value) for value in delay_range]
        if (
            len(delay_range) != 2
            or delay_range[0] < 0
            or delay_range[1] < delay_range[0]
            or delay_range[1] >= int(self.cfg.decimation)
        ):
            raise ValueError(
                "ctrl.delay_range must be inclusive [lo, hi] "
                "with 0 <= lo <= hi < decimation; got "
                f"{delay_range} for decimation={self.cfg.decimation}"
            )
        delay_mode = str(
            getattr(self.cfg.ctrl, "delay_mode", "per_episode")
        )
        if delay_mode not in ("per_episode", "per_step"):
            raise ValueError(
                "ctrl.delay_mode must be 'per_episode' or "
                f"'per_step', got {delay_mode!r}"
            )
        self._target_update_delay_bounds = (delay_range[0], delay_range[1])
        self._target_update_delay_sampling_mode = delay_mode
        self._target_update_delay_randomized = delay_range[0] != delay_range[1]
        self._target_update_delay_steps = torch.full(
            (self.num_envs,), delay_range[0], device=self.device, dtype=torch.long
        )
        self.packet_stiffness_override = None
        self.packet_deriv_override = None
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Observation history: only the proprioceptive part is stacked over the last
        # `obs_history_length` frames (concatenated oldest->newest) so the network
        # can infer velocities/dynamics from the past. The lookahead future errors
        # controller gains, and the critic's privileged dims are forward-looking /
        # episode-constant, so they are appended once from the current frame rather
        # than duplicated across the history. H = 1 keeps the single-step observation.
        self.obs_history_length = max(1, int(self.cfg.obs_history_length))
        # ee_pos(3)+ee_quat(4)+optional joint_pos(7)+optional joint_vel(7),
        # plus the (possibly gain-augmented) action vector fed back into the
        # proprio observation.
        self.action_dim = self.actions.shape[-1]

        # Command dead time. Slot d holds the action emitted d control steps ago.
        rand_cfg = self.cfg.randomization
        self._action_latency_enabled = bool(
            getattr(rand_cfg, "enable_action_latency", False)
        )
        lat_range = [int(v) for v in getattr(rand_cfg, "action_latency_range", [0, 0])]
        if len(lat_range) != 2 or lat_range[0] < 0 or lat_range[1] < lat_range[0]:
            raise ValueError(
                "randomization.action_latency_range must be [lo, hi] with "
                f"0 <= lo <= hi in whole control steps, got {lat_range}"
            )
        self._action_latency_bounds = (lat_range[0], lat_range[1])
        self._action_latency_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        if self._action_latency_enabled and lat_range[1] > 0:
            self._action_latency_buf = torch.zeros(
                (lat_range[1] + 1, self.num_envs, self.action_dim), device=self.device
            )
        else:
            self._action_latency_buf = None
        self.include_joint_angles = bool(self.cfg.include_joint_angles)
        self.include_joint_velocities = bool(self.cfg.include_joint_velocities)
        self.controller_context_in_policy = bool(self.cfg.controller_context_in_policy)
        self.controller_context_in_critic = bool(self.cfg.controller_context_in_critic)
        self.proprio_dim = (
            7
            + (7 if self.include_joint_angles else 0)
            + (7 if self.include_joint_velocities else 0)
            + self.action_dim
        )
        self.future_dim = 6 * self.cfg.tracking.num_future_steps
        # Force-tracking add-on uses the same start offset as the pose-reference
        # window and is appended to the policy/critic observation when enabled.
        self.enable_force = bool(self.cfg.tracking.enable_force)
        self.force_mode = str(getattr(self.cfg.tracking, "force_mode", "replay_disturbance"))
        self.dataset_wrench_convention = str(
            getattr(self.cfg.tracking, "dataset_wrench_convention", "world_external")
        )
        self.virtual_contact_enabled = self.enable_force and self.force_mode == "virtual_contact"
        self.wrench_dim = 6 if bool(self.cfg.tracking.use_full_wrench) else 3
        self.force_dim = self.wrench_dim * self.cfg.tracking.num_future_steps if self.enable_force else 0
        self.force_feedback_dim = self.wrench_dim if self.virtual_contact_enabled else 0
        self.proprio_dim += self.force_feedback_dim
        self._force_mag_max = max(float(self.cfg.tracking.force_mag_range[1]), 1.0e-6)
        self._torque_mag_max = max(float(self.cfg.tracking.torque_mag_max), 1.0e-6)
        wrench_scale = [self._force_mag_max] * 3
        if self.wrench_dim == 6:
            wrench_scale += [self._torque_mag_max] * 3
        self._wrench_scale = torch.tensor(wrench_scale, device=self.device)
        self.controller_context_dim = 6 if self.controller_context_in_policy else 0
        self.critic_controller_context_dim = (
            (6 if self.controller_context_in_policy else 12)
            if self.controller_context_in_critic
            else 0
        )
        self.virtual_contact_privileged_dim = 6 if self.virtual_contact_enabled else 0
        self.privileged_dim = 24 + self.virtual_contact_privileged_dim
        self.proprio_history = torch.zeros(
            (self.num_envs, self.obs_history_length, self.proprio_dim), device=self.device
        )

        self.command_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.reference_decimation = int(self.cfg.tracking.reference_decimation)
        self.policy_steps_per_reference = (
            self.reference_decimation // int(self.cfg.decimation)
        )
        self._dataset_async_reset_enabled = bool(
            self.cfg.tracking.dataset_async_reset
        )
        self.reference_dt = self.reference_decimation * float(self.cfg.sim.dt)
        self._reference_episode_steps = math.ceil(
            self.max_episode_length / self.policy_steps_per_reference
        )
        reference_buffer_duration_s = float(
            getattr(self.cfg.tracking, "reference_buffer_duration_s", 0.0)
        )
        reference_buffer_steps = math.ceil(
            reference_buffer_duration_s / self.reference_dt
        )
        self._reference_buffer_steps = max(
            self._reference_episode_steps, reference_buffer_steps
        )
        self._physics_substep_in_policy = 0

        # SONIC-style online curriculum. Analytic circles use bins on their one
        # continuous clock. Dataset references initialize flattened
        # (trajectory, local-frame-bin) metadata after their variable lengths
        # are loaded below.
        self._adaptive_phase_enabled = (
            (
                self.cfg.tracking.mode == "circle"
                and self.cfg.tracking.circle_start_time_sampling == "adaptive"
            )
            or (
                self.cfg.tracking.mode == "dataset"
                and self.cfg.tracking.dataset_chunk_start_sampling == "adaptive"
            )
        )
        adaptive_init_count = float(self.cfg.tracking.adaptive_phase_init_count)
        self._adaptive_phase_num_bins = 0
        self._adaptive_phase_visit_counts = torch.empty(0, device=self.device)
        self._adaptive_phase_failure_counts = torch.empty(0, device=self.device)
        self._adaptive_dataset_bin_offsets = None
        self._adaptive_dataset_bins_per_sequence = None
        self._adaptive_dataset_bin_sequence_ids = None
        self._adaptive_dataset_bin_starts = None
        self._adaptive_dataset_bin_ends = None
        self._adaptive_dataset_bin_weights = None
        self._adaptive_dataset_bin_size_steps = 0
        if self._adaptive_phase_enabled and self.cfg.tracking.mode == "circle":
            adaptive_bin_size = float(
                self.cfg.tracking.adaptive_phase_bin_size_s
            )
            adaptive_reference_duration = float(
                self.cfg.tracking.circle_reference_duration_s
            )
            self._adaptive_phase_num_bins = math.ceil(
                adaptive_reference_duration / adaptive_bin_size
            )
            self._adaptive_phase_visit_counts = torch.full(
                (self._adaptive_phase_num_bins,),
                adaptive_init_count,
                device=self.device,
            )
            self._adaptive_phase_failure_counts = torch.full_like(
                self._adaptive_phase_visit_counts, adaptive_init_count
            )

        # Nominal reset EE pose (from `ctrl.reset_joints`), captured at reset and
        # used as the center of the randomized start-pose sampling box.
        self.nominal_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.nominal_start_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Per-env arm joint configuration the env resets into (the cuRobo IK solution
        # for the sampled start pose). Used as the OSC nullspace posture anchor so the
        # redundancy resolution biases toward each env's own reachable start config
        # rather than a single global `ctrl.default_dof_pos_tensor`.
        self.nominal_start_joint_pos = torch.tensor(
            self.cfg.ctrl.reset_joints, device=self.device
        ).repeat(self.num_envs, 1)

        # cuRobo IK runs in a persistent subprocess (avoids a warp/PhysX GPU clash);
        # the worker + cached kinematic frames are set up lazily on first reset.
        self._ik_proc = None
        self._ik_worker_mod = None
        self._kin_frames_cached = False
        # Constant transforms between cuRobo's `panda_hand` IK frame and the
        # configured controller/dataset tool frame.
        self.hand_in_tool_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_in_tool_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.tool_in_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tool_in_hand_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # World pose of the `panda_link0` base (fixed per env).
        self.base_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Reference-trajectory parameters, sampled per env at reset.
        self.traj_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_start_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.traj_dir = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_speed = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_length = torch.zeros((self.num_envs, 1), device=self.device)
        # Circle-mode params: orthonormal plane basis (u, v), radius, signed omega.
        self.traj_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_u[:, 0] = 1.0
        self.traj_v = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_v[:, 1] = 1.0
        self.traj_radius = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_omega = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_start_time = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_rot_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_rot_axis[:, 2] = 1.0
        self.traj_rot_speed = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_rot_angle = torch.zeros((self.num_envs, 1), device=self.device)

        # Discretized reference trajectory on its native clock. In synchronous
        # mode this is one waypoint per policy step. Async mode keeps the 15 Hz
        # dataset samples native while multiple policy steps occur between them.
        # The tail covers the strict-future observation window.
        self._traj_len = (
            self._reference_buffer_steps + self.cfg.tracking.num_future_steps
        )
        self.traj_pos_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        self.traj_quat_buf = torch.zeros((self.num_envs, self._traj_len, 4), device=self.device)
        self.traj_quat_buf[..., 0] = 1.0
        # Discretized target-wrench sequence in its configured dataset convention,
        # sampled per env at reset when force tracking is enabled. Forge wrench
        # goals remain sensor-local incoming-joint reactions in this buffer.
        self.traj_wrench_buf = torch.zeros(
            (self.num_envs, self._traj_len, self.wrench_dim), device=self.device
        )
        # Six-axis external wrench actually replayed at the wrist. Dataset values
        # may be EMA-inverted into this buffer while traj_wrench_buf retains the
        # recorded smoothed force used by observations and metrics.
        self.traj_applied_wrench_buf = torch.zeros(
            (self.num_envs, self._traj_len, 6), device=self.device
        )
        # Per-timestep dataset-conditioned virtual planes. Unlike the robot-side
        # controller stiffness, these use one fixed environment stiffness k_p.
        self.traj_surface_point_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        self.traj_surface_normal_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        self.traj_surface_normal_buf[..., 2] = 1.0
        self.traj_surface_velocity_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        self.traj_rotational_equilibrium_quat_buf = torch.zeros(
            (self.num_envs, self._traj_len, 4), device=self.device
        )
        self.traj_rotational_equilibrium_quat_buf[..., 0] = 1.0
        self.traj_rotational_equilibrium_angvel_buf = torch.zeros(
            (self.num_envs, self._traj_len, 3), device=self.device
        )
        # Dataset-conditioned target external wrench in world axes. The force
        # half places the moving plane; the torque half is coupled to the force
        # realized by that plane at runtime.
        self.traj_virtual_target_wrench_buf = torch.zeros(
            (self.num_envs, self._traj_len, self.wrench_dim), device=self.device
        )
        # Per-step contact flag (True where a contact band is active), used to split
        # tracking metrics into free-space vs contact phases.
        self.traj_contact_buf = torch.zeros((self.num_envs, self._traj_len), dtype=torch.bool, device=self.device)
        # Current-step target wrench, in the same convention as traj_wrench_buf.
        self.target_wrench = torch.zeros((self.num_envs, self.wrench_dim), device=self.device)
        self.external_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        # Whether a contact band is active this step (for free/contact metric split).
        self.current_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Virtual-contact state. The surface point/normal are expressed in the same
        # base/env-local frame as the dataset trajectory. The generated force is a
        # world/base-axis vector and is rotated into the wrist link frame only when
        # written to PhysX.
        self.virtual_surface_point = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_surface_normal = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_surface_normal[:, 2] = 1.0
        self.virtual_surface_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_rotational_equilibrium_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.virtual_rotational_equilibrium_quat[:, 0] = 1.0
        self.virtual_rotational_equilibrium_angvel = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.virtual_surface_stiffness = torch.zeros((self.num_envs, 1), device=self.device)
        self.virtual_surface_damping = torch.zeros((self.num_envs, 1), device=self.device)
        self.virtual_contact_power_exponent = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.power_law_exponent),
            device=self.device,
        )
        self.virtual_contact_reference_penetration = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.power_law_reference_penetration),
            device=self.device,
        )
        self.virtual_contact_dissipation = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.hunt_crossley_dissipation),
            device=self.device,
        )
        self.virtual_contact_surface_offset = torch.zeros(
            (self.num_envs, 1), device=self.device
        )
        self.virtual_contact_normal_tilt = torch.zeros(
            (self.num_envs, 1), device=self.device
        )
        self.virtual_contact_rotational_power_exponent = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.virtual_contact_rotational_power_exponent),
            device=self.device,
        )
        self.virtual_contact_rotational_reference_penetration = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.virtual_contact_rotational_reference_penetration),
            device=self.device,
        )
        self.virtual_contact_rotational_damping = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.virtual_contact_rotational_damping),
            device=self.device,
        )
        self.virtual_contact_rotational_coefficient = torch.full(
            (self.num_envs, 1),
            float(self.cfg.tracking.virtual_contact_rotational_stiffness),
            device=self.device,
        )
        self.virtual_surface_distance = torch.full((self.num_envs, 1), 1.0, device=self.device)
        self.virtual_contact_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_contact_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_target_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.virtual_target_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.force_sensor_raw = torch.zeros((self.num_envs, self.wrench_dim), device=self.device)
        self.force_sensor_smooth = torch.zeros((self.num_envs, self.wrench_dim), device=self.device)
        self.force_sensor_bias = torch.zeros((self.num_envs, self.wrench_dim), device=self.device)
        self.force_sensor_obs = torch.zeros((self.num_envs, self.wrench_dim), device=self.device)
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        # Dataset-reference mode: per-env index of the demonstration trajectory
        # currently being tracked (into `self._ds_pos`/`_ds_quat`/`_ds_wrench`),
        # the sampled within-episode chunk start, plus the smooth-warp vectors. A
        # zero warp vector leaves a sampled demonstration unchanged.
        self.traj_ds_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.traj_ds_start = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.traj_ds_warp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_rot = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_pos_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_rot_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_phase = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_ds_warp_cycles = torch.zeros((self.num_envs, 1), device=self.device)
        self._ds_pos = None
        self._ds_quat = None
        self._ds_wrench = None
        self._ds_joint_pos = None
        self._ds_lengths = None
        self._ds_num = 0
        self._disturbance_torque_clip_norm = math.inf
        if self.cfg.tracking.mode == "dataset":
            self._load_dataset_trajectories()
            if self._adaptive_phase_enabled:
                self._initialize_dataset_adaptive_sampling()
        self._presampled_circle = None
        if self._uses_presampled_circles():
            self._load_presampled_circles()

        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        # Previous-step selected-tool velocity, used to penalize the step-to-step
        # velocity change (acceleration) as a smoothness regularizer. Legacy names
        # are retained because these tensors also feed existing checkpoints/code.
        self.prev_fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_midpoint_linaccel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_midpoint_angaccel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_jerk_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fingertip_midpoint_jacobian = torch.zeros((self.num_envs, 6, 7), device=self.device)
        self.arm_mass_matrix = torch.eye(7, device=self.device).repeat(self.num_envs, 1, 1)
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.joint_torque = torch.zeros_like(self.joint_pos)
        self.joint_torque_commanded = torch.zeros((self.num_envs, 7), device=self.device)
        self.actuator_desired_torque = torch.zeros((self.num_envs, 7), device=self.device)
        self.applied_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            self.num_envs, 1
        )
        self.task_prop_gains = self.default_gains.clone()
        self.control_gain_damping_ratio = torch.full(
            (self.num_envs, 1),
            float(self.cfg.ctrl.control_gain_damping_ratio),
            device=self.device,
        )
        self.task_deriv_gains = (
            self.control_gain_damping_ratio
            * factory_utils.get_deriv_gains(self.task_prop_gains)
        )
        # Bounds the policy-scheduled proportional gains are mapped into when
        # ctrl.control_gains is enabled (normalized action [-1, 1] -> [min, max]).
        self.gain_min = torch.tensor(self.cfg.ctrl.task_prop_gains_min, device=self.device).repeat(self.num_envs, 1)
        self.gain_max = torch.tensor(self.cfg.ctrl.task_prop_gains_max, device=self.device).repeat(self.num_envs, 1)

        self.payload_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self.payload_com = torch.zeros((self.num_envs, 3), device=self.device)
        # Residual payload first moment h=m*r in the payload-body frame, kg*m.
        # Unlike a fitted CoM, h produces the linear gravity moment h x g and
        # remains well-conditioned when the fitted residual mass is zero.
        self.payload_first_moment = torch.zeros((self.num_envs, 3), device=self.device)
        self.gravity_vec = torch.tensor(self.cfg.sim.gravity, device=self.device).repeat(self.num_envs, 1)
        self.link_mass_scales = torch.ones((self.num_envs, self._robot.num_bodies), device=self.device)
        self.joint_friction = torch.zeros((self.num_envs, 7), device=self.device)
        # Sys-id residual gravity-compensation torque. Zero unless a replay or
        # fitter writes an identified per-joint bias.
        self.joint_torque_bias = torch.zeros((self.num_envs, 7), device=self.device)
        # Per-joint reflected rotor inertia written to PhysX (domain randomization).
        # Cached for the critic's privileged observation and logging; the OSC
        # controller stays blind to it (see `arm_armature`).
        self.joint_armature = torch.zeros((self.num_envs, 7), device=self.device)

        # Validate the mass-matrix reconstruction against PhysX once at startup (nominal parameters)
        # and once more after the first dynamics randomization (payload merged, link masses scaled).
        # PhysX refreshes its generalized mass matrix only when the simulation steps, so checks run
        # at least two sim steps after the last mass-property write.
        self._mass_matrix_checks_pending = 1
        self._dynamics_write_sim_step = 0
        self._randomized_mass_matrix_validated = False

        self._resolve_robot_indices()
        if self.cfg.plant_model_urdf is None:
            factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        else:
            self._apply_plant_inertial_model(self.cfg.plant_model_urdf)
        self._cache_default_dynamics()
        self._compute_intermediate_values()
        # Fixed rotation from the configured tool/dataset frame to the force
        # sensor child-joint frame. The latter is aligned with the sensor body in
        # the Forge Franka asset (panda_hand_joint localRot1 is identity).
        sensor_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        self.force_sensor_quat_in_tool = math_utils.quat_mul(
            math_utils.quat_inv(self.fingertip_midpoint_quat), sensor_quat_w
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_track",
                "rot_track",
                "fine_pos_track",
                "fine_rot_track",
                "ee_vel",
                "ee_accel",
                "ee_jerk",
                "ee_angular_accel",
                "ee_angular_jerk",
                "action_rate",
                "action_alternation",
                "pose_action_curvature",
                "gain_action_curvature",
                "gain_rate",
                "force_track",
                "torque_track",
                "force_over",
                "force_safe",
                "stiffness_preference",
                "joint_vel",
                "joint_limit",
            ]
        }
        # Unclipped physical Cartesian diagnostics. Keep these separate from
        # Episode_Reward: the reward terms may be hard-clipped or transformed
        # and mix linear motion with 0.1x angular motion. These accumulators
        # match the fixed-evaluation linear EE units instead.
        self._episode_ee_derivative_diagnostics = {
            "accel_sum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "accel_sqsum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "accel_max": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "jerk_sum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "jerk_sqsum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "jerk_max": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_accel_sum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_accel_sqsum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_accel_max": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_jerk_sum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_jerk_sqsum": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "angular_jerk_max": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        for prefix in (
            "residual_linear_accel",
            "residual_linear_jerk",
            "residual_angular_accel",
            "residual_angular_jerk",
        ):
            self._episode_ee_derivative_diagnostics[f"{prefix}_sum"] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self._episode_ee_derivative_diagnostics[f"{prefix}_sqsum"] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self._episode_ee_derivative_diagnostics[f"{prefix}_max"] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        # Physical derivative summaries are analytics, not part of the PPO
        # objective. Keep their reductions and episode accumulators off during
        # training; common_eval_context enables them for deterministic eval.
        self._collect_ee_derivative_analytics = bool(
            self.cfg.collect_training_ee_derivative_analytics
        )
        self._skip_training_motion_regularization = bool(
            self.cfg.skip_training_motion_regularization
        )
        if self._skip_training_motion_regularization:
            nonzero_motion_weights = {
                "ee_accel_scale": self.cfg.reward.ee_accel_scale,
                "ee_jerk_scale": self.cfg.reward.ee_jerk_scale,
                "ee_angular_accel_scale": self.cfg.reward.ee_angular_accel_scale,
                "ee_angular_jerk_scale": self.cfg.reward.ee_angular_jerk_scale,
                "action_alternation_scale": self.cfg.reward.action_alternation_scale,
                "pose_action_curvature_scale": self.cfg.reward.pose_action_curvature_scale,
                "gain_action_curvature_scale": self.cfg.reward.gain_action_curvature_scale,
            }
            nonzero_motion_weights = {
                name: value
                for name, value in nonzero_motion_weights.items()
                if float(value) != 0.0
            }
            if nonzero_motion_weights:
                raise ValueError(
                    "skip_training_motion_regularization requires zero motion "
                    f"reward weights, got {nonzero_motion_weights}"
                )

        # Per-native-reference-step tracking-error profile. Async policies receive
        # rewards at their higher action rate, but evaluation metrics remain on the
        # original reference clock so they are directly comparable to 15 Hz runs.
        self._perstep_num_bins = int(self._reference_episode_steps)
        self._perstep_pos_sum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_pos_sqsum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_rot_sum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_rot_sqsum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_count = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_log_every = int(self.cfg.log_perstep_error_episodes) * self._perstep_num_bins
        self._perstep_step_counter = 0

        # Policy-scheduled PD gain statistics: when the policy controls the gains,
        # accumulate the per-step task_prop_gains (num_envs x 6) into running
        # sum/sqsum (for mean/std) and per-dim histogram bins over [gain_min,
        # gain_max], then periodically emit scalars + a histogram figure to wandb.
        # Reuses the per-step-error logging cadence and is a no-op otherwise.
        self._gain_stats_num_dims = 6
        self._gain_hist_num_bins = 40
        self._gain_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_count = torch.zeros((), device=self.device)
        self._gain_hist = torch.zeros(
            (self._gain_stats_num_dims, self._gain_hist_num_bins), device=self.device
        )
        self._gain_step_counter = 0
        # Same gain sum/sqsum, but split by whether a contact band is active this
        # step, so the scheduled-gain mean/std can be compared between free-space
        # and contact phases (only populated when force tracking is enabled).
        self._gain_free_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_free_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_free_count = torch.zeros((), device=self.device)
        self._gain_contact_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_contact_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_contact_count = torch.zeros((), device=self.device)

        # Running rollout mean of the batch tracking error. The rl_games observer
        # only logs the *last* step's `tracking_*_error` per epoch; with synchronized
        # fixed-length episodes that samples a single episode phase and aliases the
        # per-episode reference transient into a sawtooth. Averaging over a full
        # episode-length window (aligned with the synchronized resets) removes the
        # phase dependence so the logged scalar is smooth.
        self._track_err_window = int(self._reference_episode_steps)
        self._track_err_pos_sum = torch.zeros((), device=self.device)
        self._track_err_rot_sum = torch.zeros((), device=self.device)
        self._track_err_count = 0

        # Same running-window mean, but split by whether a contact band is active
        # this step (free-space vs contact phase). Sums accumulate per-env errors
        # and successes with per-category counts so the logged scalar is the mean
        # over all (env, step) samples of that category in the window. Reset in
        # lockstep with the aggregate window above.
        self._track_free_pos_sum = torch.zeros((), device=self.device)
        self._track_free_rot_sum = torch.zeros((), device=self.device)
        self._track_free_succ_sum = torch.zeros((), device=self.device)
        self._track_free_count = torch.zeros((), device=self.device)
        self._track_contact_pos_sum = torch.zeros((), device=self.device)
        self._track_contact_rot_sum = torch.zeros((), device=self.device)
        self._track_contact_succ_sum = torch.zeros((), device=self.device)
        self._track_contact_count = torch.zeros((), device=self.device)

        # Latest per-step tracking error of env 0, cached for the video overlay.
        self._vis_pos_error_norm = None
        self._vis_rot_error_norm = None

        self.set_debug_vis(self.cfg.debug_vis or self.cfg.debug_vis_force)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_robot_indices(self):
        body_names = self._robot.body_names
        tool_frame_name = str(getattr(self.cfg, "tool_frame", "auto")).strip() or "auto"
        if tool_frame_name == "auto":
            tool_frame_name = next(
                (name for name in _AUTO_TOOL_BODY_CANDIDATES if name in body_names), None
            )
        if tool_frame_name is None:
            raise RuntimeError(f"Could not find a default Franka tool body in: {body_names}")

        virtual_spec = _VIRTUAL_TOOL_FRAMES.get(tool_frame_name)
        if virtual_spec is None:
            tool_body_name = tool_frame_name
            tool_offset_pos, tool_offset_quat = _IDENTITY_TOOL_OFFSET
        else:
            tool_body_name, tool_offset_pos, tool_offset_quat = virtual_spec

        if tool_body_name not in body_names:
            raise RuntimeError(
                f"Tool frame {tool_frame_name!r} requires body {tool_body_name!r}; "
                f"available bodies: {body_names}."
            )

        self.tool_frame_name = tool_frame_name
        self.tool_body_name = tool_body_name
        self.curobo_uses_selected_tool_frame = virtual_spec is not None
        self.tool_body_idx = body_names.index(tool_body_name)
        self.tool_pos_in_body = torch.tensor(tool_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self.tool_quat_in_body = torch.tensor(tool_offset_quat, device=self.device).repeat(self.num_envs, 1)
        # Backward-compatible alias; for a virtual frame this is its parent body.
        self.fingertip_body_idx = self.tool_body_idx

        payload_body_name = self.cfg.randomization.payload_body_name
        self.payload_body_idx = (
            body_names.index(payload_body_name) if payload_body_name in body_names else self.tool_body_idx
        )

        # Needed for base-frame IK and the legacy tool->panda_hand conversion path.
        self.hand_body_idx = body_names.index("panda_hand")
        self.base_body_idx = body_names.index("panda_link0")
        if self.tool_body_idx == self.base_body_idx:
            raise RuntimeError("tool_frame must name a non-base rigid body with a PhysX Jacobian.")

        if self.enable_force and "force_sensor" not in body_names:
            raise RuntimeError(
                f"tracking.enable_force is set but no 'force_sensor' body exists in {body_names}."
            )
        force_body_name = "force_sensor" if "force_sensor" in body_names else tool_body_name
        self.force_sensor_body_idx = body_names.index(force_body_name)

        self.arm_joint_ids = list(range(7))
        joint_names = list(self._robot.joint_names)
        self.arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.isaac_arm_joint_idx = [joint_names.index(name) for name in self.arm_joint_names]

    def _cache_default_dynamics(self):
        self.default_masses = self._robot.root_physx_view.get_masses().clone()
        self.default_inertias = self._robot.root_physx_view.get_inertias().clone()
        self.default_coms = self._robot.root_physx_view.get_coms().clone()

        # Per-link nominal inertial parameters for the controller-side mass-matrix reconstruction.
        # Body 0 (fixed base) is dropped to match the PhysX jacobian rows.
        num_links = self._robot.num_bodies - 1
        self.nominal_link_masses = self.default_masses[:, 1:].to(self.device)
        self.nominal_link_inertias = self.default_inertias[:, 1:].to(self.device).view(self.num_envs, num_links, 3, 3)
        # Link-frame offset from the current link CoM (the PhysX jacobian reference point) to the
        # nominal link CoM. Nonzero only for the payload body after a payload merge.
        self.nominal_com_offsets = torch.zeros((self.num_envs, num_links, 3), device=self.device)
        if self.cfg.ctrl.nominal_model_urdf is not None:
            masses, coms, inertias, _ = map_urdf_inertials(
                list(self._robot.body_names)[1:],
                self.cfg.ctrl.nominal_model_urdf,
                aliases={"panda_fingertip_centered": "panda_fingertip"},
            )
            dtype = self.nominal_link_masses.dtype
            self.nominal_link_masses[:] = torch.as_tensor(masses, device=self.device, dtype=dtype)
            self.nominal_link_inertias[:] = torch.as_tensor(inertias, device=self.device, dtype=dtype)
            current_coms = self.default_coms[:, 1:, :3].to(self.device)
            urdf_coms = torch.as_tensor(coms, device=self.device, dtype=dtype)
            self.nominal_com_offsets[:] = urdf_coms - current_coms
        # Nominal (config-default) armature the controller's reconstructed mass matrix uses. Cached
        # once so the controller stays blind to the per-env armature randomization written to PhysX
        # in `_randomize_dynamics` (that mismatch is a disturbance the policy must reject).
        self.arm_armature = self._robot.root_physx_view.get_dof_armatures().to(self.device)[:, 0:7]

    def _apply_plant_inertial_model(self, path: str) -> None:
        """Replace USD inertials with the selected model before caching defaults."""
        body_names = list(self._robot.body_names)
        masses_np, coms_np, inertias_np, present = map_urdf_inertials(
            body_names,
            path,
            aliases={"panda_fingertip_centered": "panda_fingertip"},
        )
        masses = self._robot.root_physx_view.get_masses()
        coms = self._robot.root_physx_view.get_coms()
        inertias = self._robot.root_physx_view.get_inertias()
        mapped_masses = torch.as_tensor(masses_np, device=masses.device, dtype=masses.dtype)
        mapped_coms = torch.as_tensor(coms_np, device=coms.device, dtype=coms.dtype)
        mapped_inertias = torch.as_tensor(
            inertias_np.reshape(len(body_names), 9),
            device=inertias.device,
            dtype=inertias.dtype,
        )
        present_t = torch.as_tensor(present, device=masses.device)
        non_base = torch.arange(len(body_names), device=masses.device) > 0
        missing = non_base & ~present_t
        masses[:, present_t] = mapped_masses[present_t]
        coms[:, present_t, :3] = mapped_coms[present_t]
        inertias[:, present_t] = mapped_inertias[present_t]
        negligible = missing | (non_base & present_t & (mapped_masses <= 0.0))
        masses[:, negligible] = 1.0e-6
        coms[:, negligible, :3] = 0.0
        inertias[:, negligible] = 0.0
        for diagonal_index in (0, 4, 8):
            inertias[:, negligible, diagonal_index] = 1.0e-9
        env_ids = torch.arange(self.scene.num_envs)
        self._robot.root_physx_view.set_masses(masses, env_ids)
        self._robot.root_physx_view.set_coms(coms, env_ids)
        self._robot.root_physx_view.set_inertias(inertias, env_ids)

    def _compute_intermediate_values(self):
        # Legacy tensor names are retained for checkpoint/downstream compatibility,
        # but they now contain the configured tool-frame state.
        body_pos_w = self._robot.data.body_pos_w[:, self.tool_body_idx]
        body_quat_w = self._robot.data.body_quat_w[:, self.tool_body_idx]
        tool_pos_w, tool_quat_w = math_utils.combine_frame_transforms(
            body_pos_w, body_quat_w, self.tool_pos_in_body, self.tool_quat_in_body
        )
        body_com_pos_w = self._robot.data.body_com_pos_w[:, self.tool_body_idx]
        body_com_linvel_w = self._robot.data.body_com_lin_vel_w[:, self.tool_body_idx]
        body_angvel_w = self._robot.data.body_com_ang_vel_w[:, self.tool_body_idx]
        com_to_tool_w = tool_pos_w - body_com_pos_w

        self.fingertip_midpoint_pos = tool_pos_w - self.scene.env_origins
        self.fingertip_midpoint_quat = tool_quat_w
        self.fingertip_midpoint_linvel = body_com_linvel_w + torch.linalg.cross(
            body_angvel_w, com_to_tool_w, dim=-1
        )
        self.fingertip_midpoint_angvel = body_angvel_w

        jacobians = self._robot.root_physx_view.get_jacobians()
        tool_jacobian = jacobians[:, self.tool_body_idx - 1, 0:6, 0:7].clone()
        angular_cols = tool_jacobian[:, 3:6, :].transpose(1, 2)
        linear_shift = torch.linalg.cross(
            angular_cols, com_to_tool_w.unsqueeze(1).expand_as(angular_cols), dim=-1
        ).transpose(1, 2)
        tool_jacobian[:, 0:3, :] += linear_shift
        self.fingertip_midpoint_jacobian = tool_jacobian
        if self.cfg.ctrl.use_gt_mass_matrix:
            # Ground-truth controller: use PhysX's own generalized mass matrix, which reflects the
            # payload merge and link-mass scaling (a perfectly-modeled controller).
            self.arm_mass_matrix = (
                self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7].to(self.device)
            )
        else:
            # The controller runs on the nominal model: rebuild the mass matrix from the default inertial
            # parameters instead of querying PhysX, which would expose the payload and link-mass scales.
            self.arm_mass_matrix = self._compute_arm_mass_matrix(
                jacobians, self.nominal_link_masses, self.nominal_link_inertias, self.nominal_com_offsets
            )
        if self._mass_matrix_checks_pending > 0 and self._sim_step_counter >= self._dynamics_write_sim_step + 2:
            self._validate_arm_mass_matrix(jacobians)
            self._mass_matrix_checks_pending -= 1
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

    def _compute_arm_mass_matrix(
        self,
        jacobians: torch.Tensor,
        link_masses: torch.Tensor,
        link_inertias: torch.Tensor,
        com_offsets: torch.Tensor,
        armature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rebuild the 7x7 arm mass matrix from per-link jacobians and inertial parameters.

        M(q) = sum_l m_l J_vc^T J_vc + J_w^T (R I R^T) J_w + diag(armature). PhysX link jacobians are
        referenced at each link's *current* CoM (world axes) and inertias are link-frame tensors about
        the CoM (verified against `get_generalized_mass_matrices`). `com_offsets` is the link-frame
        offset from the current CoM to the CoM that `link_masses`/`link_inertias` refer to — nonzero
        only for the payload body when reconstructing the nominal model after a payload merge.

        Evaluated with the cached nominal parameters this gives the mass matrix of the unrandomized
        robot, independent of the payload and link-mass randomization in PhysX.

        `armature` is the per-joint reflected rotor inertia added to the diagonal; it defaults to the
        nominal `self.arm_armature` (what the controller uses, keeping it blind to the sim-side armature
        randomization). This is a controller-side modeling term only: PhysX's `get_generalized_mass_matrices`
        is the pure rigid-body inertia and excludes the DOF armature, so the validation path passes a zero
        armature to compare the reconstruction against PhysX on equal footing.
        """
        if armature is None:
            armature = self.arm_armature
        num_links = self._robot.num_bodies - 1
        J_v = jacobians[:, :, 0:3, 0:7]
        J_w = jacobians[:, :, 3:6, 0:7]

        link_quat_w = self._robot.data.body_quat_w[:, 1:]
        R = math_utils.matrix_from_quat(link_quat_w.reshape(-1, 4)).view(self.num_envs, num_links, 3, 3)
        offset_w = (R @ com_offsets.unsqueeze(-1)).squeeze(-1)
        offset_skew = math_utils.skew_symmetric_matrix(offset_w.reshape(-1, 3)).view(
            self.num_envs, num_links, 3, 3
        )
        J_vc = J_v - offset_skew @ J_w

        inertia_w = R @ link_inertias @ R.transpose(-1, -2)
        mass_term = torch.einsum("nl,nlai,nlaj->nij", link_masses, J_vc, J_vc)
        rot_term = torch.einsum("nlai,nlab,nlbj->nij", J_w, inertia_w, J_w)
        return mass_term + rot_term + torch.diag_embed(armature)

    def _validate_arm_mass_matrix(self, jacobians: torch.Tensor):
        """Check the jacobian-based reconstruction against PhysX using the current inertial parameters.

        Rebuilding with the *current* (randomized, payload-merged) parameters must reproduce PhysX's own
        generalized mass matrix; this pins down the jacobian reference conventions and the payload merge.
        Only valid once the simulation has stepped after the last dynamics write, since
        `get_generalized_mass_matrices` does not refresh until then.
        """
        num_links = self._robot.num_bodies - 1
        masses = self._robot.root_physx_view.get_masses().to(self.device)[:, 1:]
        inertias = (
            self._robot.root_physx_view.get_inertias().to(self.device)[:, 1:].view(self.num_envs, num_links, 3, 3)
        )
        zero_offsets = torch.zeros((self.num_envs, num_links, 3), device=self.device)
        # PhysX's generalized mass matrix is the pure rigid-body joint-space inertia and does NOT
        # include the DOF armature written via `write_joint_armature_to_sim` (armature is applied
        # separately by the solver). Reconstruct without any armature so the comparison isolates the
        # jacobian/inertia conventions rather than tripping on the randomized armature diagonal.
        zero_armature = torch.zeros((self.num_envs, 7), device=self.device)
        reconstructed = self._compute_arm_mass_matrix(
            jacobians, masses, inertias, zero_offsets, armature=zero_armature
        )
        physx_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        error = (reconstructed - physx_matrix).abs().max().item()
        assert error < 0.01, (
            f"Arm mass-matrix reconstruction deviates from PhysX (max abs error {error:.4f}). "
            "Likely a convention mismatch: expected jacobians referenced at the link CoM and inertia "
            "tensors expressed in the link frame about the CoM."
        )

    def _parse_action_packet(self, action_or_packet):
        """Return (action, action_rep, compliance) for tensor or dict step input."""
        if not isinstance(action_or_packet, dict):
            return action_or_packet, str(getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose")), {}
        if "action" not in action_or_packet:
            raise KeyError("Action packet must contain an 'action' tensor.")
        action = action_or_packet["action"]
        action_rep = str(action_or_packet.get("action_rep", getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose")))
        compliance = action_or_packet.get("compliance", None) or {}
        return action, action_rep, compliance

    def _fit_action_dim(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] < self.actions.shape[-1]:
            pad = torch.zeros(
                actions.shape[0],
                self.actions.shape[-1] - actions.shape[-1],
                device=self.device,
                dtype=actions.dtype,
            )
            actions = torch.cat((actions, pad), dim=-1)
        elif actions.shape[-1] > self.actions.shape[-1]:
            actions = actions[:, : self.actions.shape[-1]]
        return actions

    def _delta_action_from_abs_ee_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        if target_pose.shape[-1] < 7:
            raise ValueError(f"action_rep='abs_ee_pose' requires at least 7 dims, got {tuple(target_pose.shape)}")
        target_pos = target_pose[:, 0:3]
        target_quat = target_pose[:, 3:7]
        target_quat = target_quat / torch.linalg.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        pose_action = torch.cat(
            (pos_error / self.pos_threshold, axis_angle_error / self.rot_threshold),
            dim=-1,
        )
        if self.cfg.ctrl.control_gains and target_pose.shape[-1] >= 13:
            pose_action = torch.cat((pose_action, target_pose[:, 7:13]), dim=-1)
        return self._fit_action_dim(pose_action)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._physics_substep_in_policy = 0
        if (
            self._target_update_delay_randomized
            and self._target_update_delay_sampling_mode == "per_step"
        ):
            self._resample_target_update_delay(self._robot._ALL_INDICES)
        self.prev_prev_actions[:] = self.prev_actions
        self.prev_actions[:] = self.actions
        actions, action_rep, compliance = self._parse_action_packet(actions)
        if action_rep not in ("rel_ee_pose", "abs_ee_pose", "delta_ee_pose"):
            raise ValueError(f"Unsupported FrankaRobustTrack action_rep={action_rep!r}")
        if action_rep == "rel_ee_pose":
            raise ValueError("FrankaRobustTrackEnv does not support action_rep='rel_ee_pose'.")
        self.current_action_rep = action_rep
        self.packet_stiffness_override = compliance.get("stiffness", None)
        self.packet_deriv_override = compliance.get("damping", None)

        actions = actions.clone().to(self.device)
        if action_rep == "abs_ee_pose":
            actions = self._delta_action_from_abs_ee_pose(actions)
        else:
            actions = self._fit_action_dim(actions)
        actions = self._apply_action_latency(actions)
        self.actions = self.cfg.ctrl.ema_factor * actions.clamp(-1.0, 1.0) + (
            1.0 - self.cfg.ctrl.ema_factor
        ) * self.actions
        if action_rep == "abs_ee_pose" or (
            action_rep == "delta_ee_pose" and self.delta_target_mode == "per_control_step"
        ):
            # Snapshot the state at the policy/control-step boundary.  _apply_action
            # runs once per decimation substep. Absolute-pose packets must always
            # hold their (threshold-clipped) target throughout those substeps;
            # otherwise the converted delta is repeatedly re-anchored to the
            # moving EE and the supposedly absolute target drifts.
            self._previous_action_target_pos.copy_(self._action_target_pos)
            self._previous_action_target_quat.copy_(self._action_target_quat)
            target_pos, target_quat = self._get_action_target_pose()
            self._action_target_pos.copy_(target_pos)
            self._action_target_quat.copy_(target_quat)

    def _apply_action_latency(self, actions: torch.Tensor) -> torch.Tensor:
        """Execute the action sampled number of control steps in the past."""
        if self._action_latency_buf is None:
            return actions
        self._action_latency_buf[1:] = self._action_latency_buf[:-1].clone()
        self._action_latency_buf[0] = actions
        index = self._action_latency_steps.view(1, -1, 1).expand(
            1, self.num_envs, actions.shape[-1]
        )
        return self._action_latency_buf.gather(0, index).squeeze(0)

    def _resample_action_latency(self, env_ids: torch.Tensor):
        """Draw per-environment integer dead time and clear prior action history."""
        if self._action_latency_buf is None:
            return
        lo, hi = self._action_latency_bounds
        self._action_latency_steps[env_ids] = torch.randint(
            lo, hi + 1, (len(env_ids),), device=self.device, dtype=torch.long
        )
        self._action_latency_buf[:, env_ids] = 0.0

    def _resample_target_update_delay(self, env_ids: torch.Tensor):
        """Draw the controller-substep delay for the selected environments."""
        lo, hi = self._target_update_delay_bounds
        if lo == hi:
            self._target_update_delay_steps[env_ids] = lo
            return
        self._target_update_delay_steps[env_ids] = torch.randint(
            lo, hi + 1, (len(env_ids),), device=self.device, dtype=torch.long
        )

    def _apply_action(self):
        self._compute_intermediate_values()
        self._update_gains_from_action()
        if self.packet_stiffness_override is not None:
            self.task_prop_gains = self.packet_stiffness_override
            if self.packet_deriv_override is not None:
                self.task_deriv_gains = self.packet_deriv_override
            else:
                self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)
        if self.enable_force and not self.virtual_contact_enabled:
            # Current step's target wrench (pre-increment episode index), applied
            # directly in replay-disturbance modes. Preserve the dataset's native
            # 15 Hz wrench samples with zero-order hold between reference ticks.
            reference_idx = self._policy_step_to_reference_index(
                self.episode_length_buf
            )
            self.target_wrench[:] = self._traj_wrench_at_index(reference_idx)
        if self.virtual_contact_enabled:
            # Evaluate contact at the end of this physics interval. On the last
            # decimation substep this is exactly the next policy boundary, which
            # DirectRLEnv exposes to `_get_rewards` after incrementing
            # `episode_length_buf`. Thus the generated contact and its reward
            # target share one reference-clock instant.
            self._update_virtual_contact_force(
                completed_physics_substeps=self._physics_substep_in_policy + 1
            )
            # Match Forge's force-sensor path: `_apply_action` runs once per
            # physics substep, so filter the synthetic incoming-joint wrench at
            # the physics rate rather than once per policy/control step. The
            # configured alpha accounts for the default 60 Hz versus Forge's
            # 120 Hz physics rate.
            self._update_virtual_contact_sensor()
        self._apply_external_wrenches()
        self._physics_substep_in_policy += 1
        if self.current_action_rep == "abs_ee_pose" or (
            self.current_action_rep == "delta_ee_pose" and self.delta_target_mode == "per_control_step"
        ):
            use_previous = self._physics_substep_in_policy <= self._target_update_delay_steps
            target_pos = torch.where(
                use_previous.unsqueeze(-1),
                self._previous_action_target_pos,
                self._action_target_pos,
            )
            target_quat = torch.where(
                use_previous.unsqueeze(-1),
                self._previous_action_target_quat,
                self._action_target_quat,
            )
        else:
            target_pos, target_quat = self._get_action_target_pose()

        if self.cfg.ctrl.backend == "factory_control":
            self._apply_factory_control(target_pos, target_quat)
        elif self.cfg.ctrl.backend == "dls_ik":
            self._apply_dls_ik(target_pos, target_quat)
        else:
            raise ValueError(f"Unsupported Franka robust track controller backend: {self.cfg.ctrl.backend}")

    def _apply_external_wrenches(self):
        """Apply the payload weight (and optional target tracking wrench) as external forces.

        Robot gravity is disabled (nominal gravity compensation is assumed perfect), so the payload
        shows up as force `m * g` and moment `h x g`, where `h=m*r` is the first moment. The legacy
        payload CoM is converted to `m*r` here only so archived configurations replay exactly. In virtual
        contact mode, the computed world-frame force and contact-coupled torque
        are applied at the wrist.

        With ``sensor_child_joint_reaction`` convention, the Forge dataset stores
        PhysX's ``get_link_incoming_joint_force()`` output. An external wrench
        produces the opposite incoming-joint reaction, so direct replay applies
        the dataset signal with a minus sign in sensor-link axes. World-external
        datasets and virtual-contact forces are rotated into link axes below.
        """
        payload_quat_w = self._robot.data.body_quat_w[:, self.payload_body_idx]
        gravity_b = math_utils.quat_apply_inverse(payload_quat_w, self.gravity_vec)
        payload_force_b = self.payload_mass * gravity_b
        payload_h_b = self.payload_mass * self.payload_com + self.payload_first_moment
        payload_torque_b = torch.cross(payload_h_b, gravity_b, dim=-1)

        if not self.enable_force:
            self._robot.set_external_force_and_torque(
                forces=payload_force_b.unsqueeze(1),
                torques=payload_torque_b.unsqueeze(1),
                body_ids=[self.payload_body_idx],
            )
            return

        fs_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        if self.virtual_contact_enabled:
            self.external_contact_wrench[:, :3] = self.virtual_contact_force
            self.external_contact_wrench[:, 3:] = self.virtual_contact_torque
        else:
            idx = self._policy_step_to_reference_index(
                self.episode_length_buf
            ).clamp(0, self._traj_len - 1)
            self.external_contact_wrench[:] = self.traj_applied_wrench_buf[
                self._env_arange, idx
            ]
        if (
            not self.virtual_contact_enabled
            and self.dataset_wrench_convention == "sensor_child_joint_reaction"
        ):
            fs_force_b = -self.external_contact_wrench[:, :3]
            fs_torque_b = -self.external_contact_wrench[:, 3:]
        else:
            fs_force_b = math_utils.quat_apply_inverse(
                fs_quat_w, self.external_contact_wrench[:, :3]
            )
            fs_torque_b = math_utils.quat_apply_inverse(
                fs_quat_w, self.external_contact_wrench[:, 3:]
            )

        forces = torch.stack([payload_force_b, fs_force_b], dim=1)
        torques = torch.zeros((self.num_envs, 2, 3), device=self.device)
        torques[:, 0] = payload_torque_b
        torques[:, 1] = fs_torque_b
        self._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[self.payload_body_idx, self.force_sensor_body_idx],
        )

    def _update_gains_from_action(self):
        """Map the gain action suffix to task PD gains when gain control is on.

        Gain actions are clamped to [-1, 1] and affinely mapped onto
        [gain_min, gain_max]; derivative gains use the per-environment damping
        ratio sampled at reset. No-op when `ctrl.control_gains` is False.
        """
        if not self.cfg.ctrl.control_gains:
            return
        gain_actions = self.actions[:, 6:].clamp(-1.0, 1.0)
        if str(self.cfg.ctrl.control_gain_action_mode) == "scalar":
            gain_actions = gain_actions.expand(-1, 6)
        normalized = 0.5 * (gain_actions + 1.0)
        self.task_prop_gains = self.gain_min + (self.gain_max - self.gain_min) * normalized
        self.task_deriv_gains = (
            self.control_gain_damping_ratio
            * factory_utils.get_deriv_gains(self.task_prop_gains)
        )

    def _get_action_target_pose(self):
        pos_actions = self.actions[:, 0:3] * self.pos_threshold
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        target_pos = self.fingertip_midpoint_pos + pos_actions

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / torch.clamp(angle.unsqueeze(-1), min=1.0e-6)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        delta_quat = torch.where(angle.unsqueeze(-1) > 1.0e-6, delta_quat, identity_quat)
        target_quat = torch_utils.quat_mul(delta_quat, self.fingertip_midpoint_quat)
        return target_pos, target_quat

    def _apply_factory_control(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            nullspace_joint_target=self.nominal_start_joint_pos,
            apply_task_inertia=self.cfg.ctrl.use_task_space_inertia,
            use_singularity_robust_inverse=(
                self.cfg.ctrl.singularity_robust_inverse
            ),
        )
        # Model residual gravity-compensation torque after OSC, before PhysX
        # applies its effort limit.
        self.joint_torque[:, 0:7] += self.joint_torque_bias
        limits = torch.as_tensor(
            self.cfg.ctrl.torque_command_limits,
            device=self.device,
            dtype=self.joint_torque.dtype,
        )
        self.joint_torque_commanded[:] = torch.clamp(
            self.joint_torque[:, 0:7], -limits, limits
        )
        if self.cfg.ctrl.apply_libfranka_torque_shaping:
            hardware_hz = float(self.cfg.ctrl.actuator_hardware_hz)
            ticks_float = self.physics_dt * hardware_hz
            ticks = int(round(ticks_float))
            if ticks < 1 or not math.isclose(ticks_float, ticks, abs_tol=1.0e-6):
                raise ValueError(
                    "libfranka torque shaping requires physics_dt * "
                    f"actuator_hardware_hz to be an integer, got {ticks_float}"
                )
            interval_torque, final_desired = shape_torque_held_command(
                self.joint_torque_commanded,
                self.actuator_desired_torque,
                ticks=ticks,
                hardware_dt=1.0 / hardware_hz,
                cutoff_hz=float(self.cfg.ctrl.actuator_filter_cutoff_hz),
                max_rate=float(self.cfg.ctrl.actuator_torque_rate_limit),
            )
            self.actuator_desired_torque[:] = final_desired
            self.joint_torque[:, 0:7] = interval_torque
        else:
            self.joint_torque[:, 0:7] = self.joint_torque_commanded
        self.ctrl_target_joint_pos[:, 7:] = 0.04
        self.joint_torque[:, 7:] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def reset_actuator_shaping(
        self, env_ids: torch.Tensor, desired_torque: torch.Tensor | None = None
    ) -> None:
        """Reset or seed the previous ``tau_J_d`` used by libfranka shaping."""
        if desired_torque is None:
            self.actuator_desired_torque[env_ids] = 0.0
        else:
            self.actuator_desired_torque[env_ids] = desired_torque

    def reset_target_schedule(self, env_ids: torch.Tensor) -> None:
        """Seed both target phases at the current controlled pose."""
        self._action_target_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self._action_target_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
        self._previous_action_target_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self._previous_action_target_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]

    def _apply_dls_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        delta_joint_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_pose,
            ik_method="dls",
            jacobian=self.fingertip_midpoint_jacobian,
            device=self.device,
        )
        self.ctrl_target_joint_pos[:] = self.joint_pos
        self.ctrl_target_joint_pos[:, 0:7] = self.joint_pos[:, 0:7] + delta_joint_pos[:, 0:7]
        self.ctrl_target_joint_pos[:, 7:] = 0.04

        self.joint_torque.zero_()
        joint_error = self.ctrl_target_joint_pos[:, 0:7] - self.joint_pos[:, 0:7]
        joint_torque = self.cfg.ctrl.joint_pos_kp * joint_error - self.cfg.ctrl.joint_pos_kd * self.joint_vel[:, 0:7]
        self.joint_torque[:, 0:7] = torch.clamp(joint_torque, min=-100.0, max=100.0)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _compute_obs_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the current-frame observation parts.

        Returns (proprio, future_errors, controller_context, privileged): `proprio`
        is stacked over history; `future_errors` and `controller_context` are fed
        once to both policy and critic; and `privileged` is critic-only information.
        """
        self._compute_intermediate_values()
        self._update_command()
        future_errors = self._future_command_errors()
        proprio_parts = [self.fingertip_midpoint_pos, self.fingertip_midpoint_quat]
        if self.include_joint_angles:
            proprio_parts.append(self.joint_pos[:, 0:7])
        if self.include_joint_velocities:
            proprio_parts.append(self.joint_vel[:, 0:7])
        if self.virtual_contact_enabled:
            noise_scale = [float(self.cfg.tracking.force_sensor_noise_std)] * 3
            if self.wrench_dim == 6:
                noise_scale += [float(self.cfg.tracking.torque_sensor_noise_std)] * 3
            sensor_noise_scale = torch.tensor(noise_scale, device=self.device)
            sensor_noise = sensor_noise_scale * torch.randn_like(self.force_sensor_smooth)
            self.force_sensor_obs = descale_virtual_sensor_wrench(
                self.force_sensor_smooth,
                self.cfg.tracking.virtual_contact_force_scale,
                self.cfg.tracking.virtual_contact_torque_scale,
            )
            self.force_sensor_obs += self.force_sensor_bias + sensor_noise
            proprio_parts.append(self.force_sensor_obs / self._wrench_scale)
        proprio_parts.append(self.actions)
        proprio = torch.cat(proprio_parts, dim=-1)
        controller_context = self.task_prop_gains
        privileged_parts = [
            self.payload_mass,
            self.payload_com,
            self.joint_friction,
            self.joint_armature,
            self.pos_threshold,
            self.rot_threshold,
        ]
        if self.virtual_contact_enabled:
            if str(self.cfg.tracking.contact_model) == "power_law":
                if bool(self.cfg.tracking.virtual_contact_randomize_parameters):
                    exponent_max = float(
                        self.cfg.tracking.power_law_exponent_range[1]
                    )
                    penetration_min = float(
                        self.cfg.tracking.power_law_reference_penetration_range[0]
                    )
                    stiffness_max = float(
                        self.cfg.tracking.power_law_reference_force
                    ) / penetration_min**exponent_max
                else:
                    stiffness_max = max(
                        float(self.cfg.tracking.power_law_coefficient), 1.0e-6
                    )
            else:
                stiffness_max = max(float(self.cfg.tracking.virtual_contact_plane_stiffness), 1.0e-6)
            damping_max = 2.0 * max(float(self.cfg.tracking.virtual_contact_damping_ratio), 1.0e-6) * math.sqrt(
                stiffness_max * max(float(self.cfg.tracking.virtual_contact_effective_mass), 1.0e-6)
            )
            privileged_parts.extend(
                [
                    self.virtual_surface_normal,
                    self.virtual_surface_distance / 0.01,
                    self.virtual_surface_stiffness / stiffness_max,
                    self.virtual_surface_damping / max(damping_max, 1.0e-6),
                ]
            )
        privileged = torch.cat(privileged_parts, dim=-1)
        return proprio, future_errors, controller_context, privileged

    def _get_observations(self) -> dict:
        proprio, future_errors, controller_context, privileged = self._compute_obs_parts()
        policy_controller_context = (
            controller_context
            if self.controller_context_in_policy
            else controller_context[:, :0]
        )
        if self.controller_context_in_critic:
            # The critic already inherits Kp through the policy observation when
            # controller context is actor-visible. Otherwise provide both Kp and
            # Kd as critic-only privileged state. Gains are sampled at reset and
            # stay fixed for the episode.
            critic_controller_context = (
                self.task_deriv_gains
                if self.controller_context_in_policy
                else torch.cat((self.task_prop_gains, self.task_deriv_gains), dim=-1)
            )
        else:
            critic_controller_context = controller_context[:, :0]
        if self.obs_history_length == 1:
            proprio_stacked = proprio
        else:
            # Shift the proprio history left by one frame and append the newest
            # frame, then flatten oldest->newest into a single vector per env.
            self.proprio_history = torch.roll(self.proprio_history, shifts=-1, dims=1)
            self.proprio_history[:, -1] = proprio
            proprio_stacked = self.proprio_history.reshape(self.num_envs, -1)

        if self.enable_force:
            future_wrench = self._future_wrench()
            policy_obs = torch.cat(
                [proprio_stacked, future_errors, future_wrench, policy_controller_context], dim=-1
            )
            critic_obs = torch.cat(
                [
                    proprio_stacked,
                    future_errors,
                    future_wrench,
                    policy_controller_context,
                    critic_controller_context,
                    privileged,
                ],
                dim=-1,
            )
        else:
            future_wrench = None
            policy_obs = torch.cat(
                [proprio_stacked, future_errors, policy_controller_context], dim=-1
            )
            critic_obs = torch.cat(
                [
                    proprio_stacked,
                    future_errors,
                    policy_controller_context,
                    critic_controller_context,
                    privileged,
                ],
                dim=-1,
            )

        # Fail loud if a NaN/Inf reaches the policy/critic input. If this fires, the
        # crash is env-side (controller / sim state) rather than the policy std
        # diverging on its own; the named part tells you where the NaN entered.
        parts = {
            "fingertip_midpoint_pos": self.fingertip_midpoint_pos,
            "fingertip_midpoint_quat": self.fingertip_midpoint_quat,
            "joint_pos": self.joint_pos[:, 0:7],
            "actions": self.actions,
            "future_errors": future_errors,
            "controller_context": controller_context,
            "task_deriv_gains": self.task_deriv_gains,
            "privileged": privileged,
        }
        if future_wrench is not None:
            parts["future_wrench"] = future_wrench
        if self.virtual_contact_enabled:
            parts["force_sensor_obs"] = self.force_sensor_obs
        if self.cfg.validate_observations:
            bad = {k: int((~torch.isfinite(v)).any(dim=-1).sum()) for k, v in parts.items()}
            if any(bad.values()):
                raise RuntimeError(
                    f"Non-finite values in observation before feeding the policy: "
                    f"{ {k: n for k, n in bad.items() if n} } (per-part env counts). "
                    f"NaN originates in the env/controller, not the policy std."
                )
        if policy_obs.shape[-1] != int(self.cfg.observation_space) or critic_obs.shape[-1] != int(
            self.cfg.state_space
        ):
            raise RuntimeError(
                "RobustTrack observation-space mismatch: "
                f"policy tensor/config={policy_obs.shape[-1]}/{self.cfg.observation_space}, "
                f"critic tensor/config={critic_obs.shape[-1]}/{self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_command()
        pos_error, rot_error = self._command_errors()
        raw_pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
        if self.virtual_contact_enabled:
            normal_error = (pos_error * self.virtual_surface_normal).sum(dim=-1)
            tangent_error = pos_error - normal_error.unsqueeze(-1) * self.virtual_surface_normal
            tangent_error_norm = torch.linalg.norm(tangent_error, dim=-1)
            near_temp = max(float(self.cfg.reward.stiffness_near_temperature), 1.0e-6)
            near_contact = torch.sigmoid(
                (
                    float(self.cfg.reward.stiffness_near_margin)
                    - self.virtual_surface_distance.squeeze(-1)
                )
                / near_temp
            )
            contact_normal_weight = float(self.cfg.reward.contact_normal_pose_weight)
            normal_pose_weight = 1.0 - near_contact * (1.0 - contact_normal_weight)
            pos_error_norm = torch.sqrt(
                tangent_error_norm.square() + (normal_pose_weight * normal_error).square()
            )
        else:
            near_contact = torch.zeros(self.num_envs, device=self.device)
            pos_error_norm = raw_pos_error_norm
        rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
        ee_vel_norm = torch.linalg.norm(self.fingertip_midpoint_linvel, dim=-1) + 0.1 * torch.linalg.norm(
            self.fingertip_midpoint_angvel, dim=-1
        )
        # These finite differences shrink when the policy period is shortened.
        # Normalize them back to the native-reference (15 Hz) interval so the
        # action-rate coefficient retains its established meaning. EE motion
        # derivatives use the physical policy timestep for new configurations.
        rate_normalization = float(self.policy_steps_per_reference)
        derivative_mode = str(self.cfg.reward.ee_derivative_mode)
        derivative_penalty_mode = str(self.cfg.reward.ee_derivative_penalty_mode)
        difference_interval = (
            float(self.step_dt)
            if derivative_mode == "physical"
            else 1.0 / rate_normalization
        )
        acceleration_clip_value = float(self.cfg.reward.ee_accel_clip)
        jerk_clip_value = float(self.cfg.reward.ee_jerk_clip)
        acceleration_clip = (
            acceleration_clip_value
            if derivative_mode == "physical"
            and derivative_penalty_mode != "residual_huber"
            and acceleration_clip_value > 0.0
            else None
        )
        jerk_clip = (
            jerk_clip_value
            if derivative_mode == "physical"
            and derivative_penalty_mode != "residual_huber"
            and jerk_clip_value > 0.0
            else None
        )
        base_derivative_penalty_mode = (
            "hard_clip" if derivative_penalty_mode == "residual_huber" else derivative_penalty_mode
        )
        zero_motion_cost = torch.zeros(self.num_envs, device=self.device)
        if (
            self._skip_training_motion_regularization
            and not self._collect_ee_derivative_analytics
        ):
            ee_accel_norm = zero_motion_cost
            ee_jerk_norm = zero_motion_cost
            current_linear_acceleration = (
                self.fingertip_midpoint_linvel
                - self.prev_fingertip_midpoint_linvel
            ) / difference_interval
            current_angular_acceleration = (
                self.fingertip_midpoint_angvel
                - self.prev_fingertip_midpoint_angvel
            ) / difference_interval
        else:
            (
                ee_accel_norm,
                ee_jerk_norm,
                current_linear_acceleration,
                current_angular_acceleration,
            ) = ee_acceleration_and_jerk_norms(
                self.fingertip_midpoint_linvel,
                self.fingertip_midpoint_angvel,
                self.prev_fingertip_midpoint_linvel,
                self.prev_fingertip_midpoint_angvel,
                self.prev_fingertip_midpoint_linaccel,
                self.prev_fingertip_midpoint_angaccel,
                difference_interval,
                self.ee_jerk_initialized,
                acceleration_clip=acceleration_clip,
                jerk_clip=jerk_clip,
                penalty_mode=base_derivative_penalty_mode,
            )
        raw_linear_accel_norm = zero_motion_cost
        raw_linear_jerk_norm = zero_motion_cost
        raw_angular_accel_norm = zero_motion_cost
        raw_angular_jerk_norm = zero_motion_cost
        if self._collect_ee_derivative_analytics:
            raw_linear_accel_norm, raw_linear_jerk_norm = (
                raw_linear_acceleration_and_jerk_norms(
                    current_linear_acceleration,
                    self.prev_fingertip_midpoint_linaccel,
                    difference_interval,
                    self.ee_jerk_initialized,
                )
            )
            raw_angular_accel_norm, raw_angular_jerk_norm = (
                raw_angular_acceleration_and_jerk_norms(
                    current_angular_acceleration,
                    self.prev_fingertip_midpoint_angaccel,
                    difference_interval,
                    self.ee_jerk_initialized,
                )
            )
        ee_angular_accel_cost = zero_motion_cost
        ee_angular_jerk_cost = zero_motion_cost
        residual_linear_accel_norm = zero_motion_cost
        residual_linear_jerk_norm = zero_motion_cost
        residual_angular_accel_norm = zero_motion_cost
        residual_angular_jerk_norm = zero_motion_cost
        if derivative_penalty_mode == "residual_huber":
            (
                reference_linear_acceleration,
                reference_angular_acceleration,
                previous_reference_linear_acceleration,
                previous_reference_angular_acceleration,
            ) = self._reference_accelerations_at_policy_step(
                self.episode_length_buf,
                difference_interval,
            )
            (
                ee_accel_norm,
                ee_jerk_norm,
                ee_angular_accel_cost,
                ee_angular_jerk_cost,
                residual_linear_accel_norm,
                residual_linear_jerk_norm,
                residual_angular_accel_norm,
                residual_angular_jerk_norm,
            ) = residual_huber_motion_costs(
                current_linear_acceleration,
                current_angular_acceleration,
                self.prev_fingertip_midpoint_linaccel,
                self.prev_fingertip_midpoint_angaccel,
                reference_linear_acceleration,
                reference_angular_acceleration,
                previous_reference_linear_acceleration,
                previous_reference_angular_acceleration,
                difference_interval,
                self.ee_jerk_initialized,
                linear_acceleration_normalizer=acceleration_clip_value,
                linear_jerk_normalizer=jerk_clip_value,
                angular_acceleration_normalizer=(
                    self.cfg.reward.ee_angular_accel_huber_normalizer
                ),
                angular_jerk_normalizer=(
                    self.cfg.reward.ee_angular_jerk_huber_normalizer
                ),
            )
        # Per-policy-step scalars consumed only by deterministic eval. Unlike
        # Episode_Reward, these remain physical and untransformed.
        if self._collect_ee_derivative_analytics:
            self.extras["ee_linear_accel_mean"] = raw_linear_accel_norm.mean()
            self.extras["ee_linear_accel_sqmean"] = raw_linear_accel_norm.square().mean()
            self.extras["ee_linear_accel_max"] = raw_linear_accel_norm.max()
            self.extras["ee_linear_jerk_mean"] = raw_linear_jerk_norm.mean()
            self.extras["ee_linear_jerk_sqmean"] = raw_linear_jerk_norm.square().mean()
            self.extras["ee_linear_jerk_max"] = raw_linear_jerk_norm.max()
            self.extras["ee_angular_accel_mean"] = raw_angular_accel_norm.mean()
            self.extras["ee_angular_accel_sqmean"] = raw_angular_accel_norm.square().mean()
            self.extras["ee_angular_accel_max"] = raw_angular_accel_norm.max()
            self.extras["ee_angular_jerk_mean"] = raw_angular_jerk_norm.mean()
            self.extras["ee_angular_jerk_sqmean"] = raw_angular_jerk_norm.square().mean()
            self.extras["ee_angular_jerk_max"] = raw_angular_jerk_norm.max()
            self.extras["ee_residual_linear_accel_mean"] = residual_linear_accel_norm.mean()
            self.extras["ee_residual_linear_jerk_mean"] = residual_linear_jerk_norm.mean()
            self.extras["ee_residual_angular_accel_mean"] = residual_angular_accel_norm.mean()
            self.extras["ee_residual_angular_jerk_mean"] = residual_angular_jerk_norm.mean()
        if (
            self._collect_ee_derivative_analytics
            and derivative_penalty_mode == "residual_huber"
        ):
            quantiles = torch.tensor((0.5, 0.75, 0.9, 0.99), device=self.device)
            residual_names = (
                "linear_accel",
                "linear_jerk",
                "angular_accel",
                "angular_jerk",
            )
            residual_samples = torch.stack(
                (
                    residual_linear_accel_norm,
                    residual_linear_jerk_norm,
                    residual_angular_accel_norm,
                    residual_angular_jerk_norm,
                )
            )
            percentile_values = torch.quantile(residual_samples, quantiles, dim=1)
            for percentile_index, percentile in enumerate((50, 75, 90, 99)):
                for residual_index, name in enumerate(residual_names):
                    self.extras[f"ee_residual_{name}_p{percentile}"] = (
                        percentile_values[percentile_index, residual_index]
                    )
        self.prev_fingertip_midpoint_linvel = self.fingertip_midpoint_linvel.clone()
        self.prev_fingertip_midpoint_angvel = self.fingertip_midpoint_angvel.clone()
        self.prev_fingertip_midpoint_linaccel = current_linear_acceleration
        self.prev_fingertip_midpoint_angaccel = current_angular_acceleration
        self.ee_jerk_initialized.fill_(True)
        # Cartesian delta-pose action rate uses only the first 6 dims so it is
        # unaffected by the optional gain dims, which get their own penalty.
        action_rate = rate_normalization * torch.linalg.norm(
            self.actions[:, 0:6] - self.prev_actions[:, 0:6], dim=-1
        )
        if (
            float(self.cfg.reward.action_alternation_scale) != 0.0
            or self._collect_ee_derivative_analytics
        ):
            action_alternation = action_alternation_penalty(
                self.actions, self.prev_actions, self.prev_prev_actions
            )
        else:
            action_alternation = zero_motion_cost
        curvature_ready = self.episode_length_buf >= 2
        collect_pose_curvature = (
            float(self.cfg.reward.pose_action_curvature_scale) != 0.0
            or self._collect_ee_derivative_analytics
        )
        if collect_pose_curvature:
            pose_action_curvature = action_curvature_huber_penalty(
                self.actions,
                self.prev_actions,
                self.prev_prev_actions,
                start=0,
                stop=6,
                normalizer=(
                    self.cfg.reward.pose_action_curvature_normalizer
                    / rate_normalization**2
                ),
            )
            pose_action_curvature = torch.where(
                curvature_ready,
                pose_action_curvature,
                torch.zeros_like(pose_action_curvature),
            )
        else:
            pose_action_curvature = zero_motion_cost
        if self.cfg.ctrl.control_gains:
            gain_rate = rate_normalization * torch.linalg.norm(
                self.actions[:, 6:12] - self.prev_actions[:, 6:12], dim=-1
            )
            collect_gain_curvature = (
                float(self.cfg.reward.gain_action_curvature_scale) != 0.0
                or self._collect_ee_derivative_analytics
            )
            if collect_gain_curvature:
                gain_action_curvature = action_curvature_huber_penalty(
                    self.actions,
                    self.prev_actions,
                    self.prev_prev_actions,
                    start=6,
                    stop=None,
                    normalizer=(
                        self.cfg.reward.gain_action_curvature_normalizer
                        / rate_normalization**2
                    ),
                )
                gain_action_curvature = torch.where(
                    curvature_ready,
                    gain_action_curvature,
                    torch.zeros_like(gain_action_curvature),
                )
            else:
                gain_action_curvature = zero_motion_cost
        else:
            gain_rate = torch.zeros(self.num_envs, device=self.device)
            gain_action_curvature = torch.zeros(self.num_envs, device=self.device)
        if self._collect_ee_derivative_analytics:
            self.extras["pose_action_curvature_cost_mean"] = pose_action_curvature.mean()
            self.extras["pose_action_curvature_cost_max"] = pose_action_curvature.max()
            self.extras["gain_action_curvature_cost_mean"] = gain_action_curvature.mean()
            self.extras["gain_action_curvature_cost_max"] = gain_action_curvature.max()
        joint_vel_norm = torch.linalg.norm(self.joint_vel[:, 0:7], dim=-1)
        joint_limit_penalty = self._joint_limit_penalty()

        force_track = torch.zeros(self.num_envs, device=self.device)
        torque_track = torch.zeros_like(force_track)
        force_over = torch.zeros_like(force_track)
        force_safe = torch.zeros_like(force_track)
        stiffness_preference = torch.zeros_like(force_track)
        force_error = torch.zeros_like(force_track)
        torque_error = torch.zeros_like(force_track)
        if self.virtual_contact_enabled:
            goal_force = self.target_wrench[:, :3]
            goal_torque = self.target_wrench[:, 3:]
            measured_wrench = descale_virtual_sensor_wrench(
                self.force_sensor_smooth,
                self.cfg.tracking.virtual_contact_force_scale,
                self.cfg.tracking.virtual_contact_torque_scale,
            )
            measured_force = measured_wrench[:, :3]
            measured_torque = measured_wrench[:, 3:]
            goal_force_mag = torch.linalg.norm(goal_force, dim=-1)
            actual_force_mag = torch.linalg.norm(self.virtual_contact_force, dim=-1)
            goal_torque_mag = torch.linalg.norm(goal_torque, dim=-1)
            actual_torque_mag = torch.linalg.norm(self.virtual_contact_torque, dim=-1)
            # Match the complete force vector in the dataset frame in both
            # force-only and full-wrench modes.
            force_error = torch.linalg.norm(measured_force - goal_force, dim=-1)
            force_track = (
                torch.exp(-force_error / max(float(self.cfg.reward.force_error_temp), 1.0e-6))
                * self.cfg.reward.force_error_scale
            )
            if self.wrench_dim == 6:
                torque_error = torch.linalg.norm(measured_torque - goal_torque, dim=-1)
                torque_track = (
                    torch.exp(
                        -torque_error
                        / max(float(self.cfg.reward.torque_error_temp), 1.0e-6)
                    )
                    * self.cfg.reward.torque_error_scale
                )
            force_over = (
                torch.relu(
                    actual_force_mag
                    - goal_force_mag
                    - float(self.cfg.reward.force_over_margin)
                ).square()
                * self.cfg.reward.force_over_scale
            )
            force_safe = (
                torch.relu(actual_force_mag - float(self.cfg.reward.force_safe_threshold)).square()
                * self.cfg.reward.force_safe_scale
            )

            normal_sq = self.virtual_surface_normal.square()
            normal_gain = (normal_sq * self.task_prop_gains[:, :3]).sum(dim=-1)
            if self.cfg.ctrl.control_gains:
                # This term trains the policy's gain-action dimensions. With fixed
                # gains it is not actionable and would instead distort the pose
                # objective through its dependence on `near_contact`.
                low_gain = (normal_sq * self.gain_min[:, :3]).sum(dim=-1)
                high_gain = (normal_sq * self.gain_max[:, :3]).sum(dim=-1)
                preferred_gain = high_gain + near_contact * (low_gain - high_gain)
                normalized_gain_error = (normal_gain - preferred_gain) / (high_gain - low_gain).clamp(min=1.0e-6)
                stiffness_preference = (
                    normalized_gain_error.square() * self.cfg.reward.stiffness_preference_scale
                )

            self.extras["force_error"] = force_error.mean()
            self.extras["force_goal_mean"] = goal_force_mag.mean()
            self.extras["force_measured_mean"] = actual_force_mag.mean()
            if self.wrench_dim == 6:
                self.extras["torque_error"] = torque_error.mean()
                self.extras["torque_goal_mean"] = goal_torque_mag.mean()
                self.extras["torque_measured_mean"] = actual_torque_mag.mean()
            self.extras["force_over_goal_fraction"] = (
                actual_force_mag > goal_force_mag + float(self.cfg.reward.force_over_margin)
            ).float().mean()
            self.extras["normal_stiffness_mean"] = normal_gain.mean()
            if bool(self.cfg.tracking.virtual_contact_randomize_parameters):
                self.extras["contact_power_exponent_mean"] = (
                    self.virtual_contact_power_exponent.mean()
                )
                self.extras["contact_reference_penetration_mean_mm"] = (
                    1000.0 * self.virtual_contact_reference_penetration.mean()
                )
                self.extras["contact_dissipation_mean"] = (
                    self.virtual_contact_dissipation.mean()
                )
                self.extras["contact_surface_offset_mean_mm"] = (
                    1000.0 * self.virtual_contact_surface_offset.mean()
                )
                self.extras["contact_normal_tilt_mean_deg"] = (
                    (180.0 / math.pi) * self.virtual_contact_normal_tilt.mean()
                )
                if self.wrench_dim == 6:
                    self.extras["contact_rotational_power_exponent_mean"] = (
                        self.virtual_contact_rotational_power_exponent.mean()
                    )
                    self.extras["contact_rotational_penetration_mean"] = (
                        self.virtual_contact_rotational_reference_penetration.mean()
                    )
                    self.extras["contact_rotational_damping_mean"] = (
                        self.virtual_contact_rotational_damping.mean()
                    )
            elif bool(self.cfg.tracking.virtual_contact_randomize_surface_offset):
                self.extras["contact_surface_offset_mean_mm"] = (
                    1000.0 * self.virtual_contact_surface_offset.mean()
                )

        startup_smoothness_multiplier = 1.0 + (
            self.cfg.reward.startup_smoothness_multiplier - 1.0
        ) * (
            self.episode_length_buf < self.cfg.reward.startup_smoothness_steps
        ).float(
        )
        rewards = {
            "pos_track": torch.exp(-pos_error_norm / self.cfg.reward.pos_error_temp) * self.cfg.reward.pos_error_scale,
            "rot_track": torch.exp(-rot_error_norm / self.cfg.reward.rot_error_temp) * self.cfg.reward.rot_error_scale,
            "fine_pos_track": (
                torch.exp(-pos_error_norm / self.cfg.reward.fine_pos_error_temp)
                * self.cfg.reward.fine_pos_error_scale
            ),
            "fine_rot_track": (
                torch.exp(-rot_error_norm / self.cfg.reward.fine_rot_error_temp)
                * self.cfg.reward.fine_rot_error_scale
            ),
            "ee_vel": ee_vel_norm * self.cfg.reward.ee_vel_scale,
            "ee_accel": (
                ee_accel_norm
                * self.cfg.reward.ee_accel_scale
                * startup_smoothness_multiplier
            ),
            "ee_jerk": (
                ee_jerk_norm
                * self.cfg.reward.ee_jerk_scale
                * startup_smoothness_multiplier
            ),
            "ee_angular_accel": (
                ee_angular_accel_cost * self.cfg.reward.ee_angular_accel_scale
            ),
            "ee_angular_jerk": (
                ee_angular_jerk_cost * self.cfg.reward.ee_angular_jerk_scale
            ),
            "action_rate": (
                action_rate
                * self.cfg.reward.action_rate_scale
                * startup_smoothness_multiplier
            ),
            "action_alternation": (
                action_alternation * self.cfg.reward.action_alternation_scale
            ),
            "pose_action_curvature": (
                pose_action_curvature * self.cfg.reward.pose_action_curvature_scale
            ),
            "gain_action_curvature": (
                gain_action_curvature * self.cfg.reward.gain_action_curvature_scale
            ),
            "gain_rate": gain_rate * self.cfg.reward.gain_rate_scale,
            "force_track": force_track,
            "torque_track": torque_track,
            "force_over": force_over,
            "force_safe": force_safe,
            "stiffness_preference": stiffness_preference,
            "joint_vel": joint_vel_norm * self.cfg.reward.joint_vel_scale,
            "joint_limit": joint_limit_penalty * self.cfg.reward.joint_limit_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) * self.step_dt
        for key, value in rewards.items():
            self._episode_sums[key] += value * self.step_dt
        if self._collect_ee_derivative_analytics:
            derivative_diagnostics = self._episode_ee_derivative_diagnostics
            derivative_diagnostics["accel_sum"] += raw_linear_accel_norm * self.step_dt
            derivative_diagnostics["accel_sqsum"] += (
                raw_linear_accel_norm.square() * self.step_dt
            )
            derivative_diagnostics["accel_max"] = torch.maximum(
                derivative_diagnostics["accel_max"], raw_linear_accel_norm
            )
            derivative_diagnostics["jerk_sum"] += raw_linear_jerk_norm * self.step_dt
            derivative_diagnostics["jerk_sqsum"] += (
                raw_linear_jerk_norm.square() * self.step_dt
            )
            derivative_diagnostics["jerk_max"] = torch.maximum(
                derivative_diagnostics["jerk_max"], raw_linear_jerk_norm
            )
            derivative_diagnostics["angular_accel_sum"] += (
                raw_angular_accel_norm * self.step_dt
            )
            derivative_diagnostics["angular_accel_sqsum"] += (
                raw_angular_accel_norm.square() * self.step_dt
            )
            derivative_diagnostics["angular_accel_max"] = torch.maximum(
                derivative_diagnostics["angular_accel_max"], raw_angular_accel_norm
            )
            derivative_diagnostics["angular_jerk_sum"] += (
                raw_angular_jerk_norm * self.step_dt
            )
            derivative_diagnostics["angular_jerk_sqsum"] += (
                raw_angular_jerk_norm.square() * self.step_dt
            )
            derivative_diagnostics["angular_jerk_max"] = torch.maximum(
                derivative_diagnostics["angular_jerk_max"], raw_angular_jerk_norm
            )
            for prefix, value in (
                ("residual_linear_accel", residual_linear_accel_norm),
                ("residual_linear_jerk", residual_linear_jerk_norm),
                ("residual_angular_accel", residual_angular_accel_norm),
                ("residual_angular_jerk", residual_angular_jerk_norm),
            ):
                derivative_diagnostics[f"{prefix}_sum"] += value * self.step_dt
                derivative_diagnostics[f"{prefix}_sqsum"] += (
                    value.square() * self.step_dt
                )
                derivative_diagnostics[f"{prefix}_max"] = torch.maximum(
                    derivative_diagnostics[f"{prefix}_max"], value
                )

        successes = torch.logical_and(
            pos_error_norm < self.cfg.reward.success_pos_threshold,
            rot_error_norm < self.cfg.reward.success_rot_threshold,
        )
        if self.virtual_contact_enabled:
            successes = torch.logical_and(
                successes,
                force_error < float(self.cfg.reward.force_success_threshold),
            )
            if self.wrench_dim == 6:
                successes = torch.logical_and(
                    successes,
                    torque_error < float(self.cfg.reward.torque_success_threshold),
                )
        native_metric_mask = is_reference_boundary(
            self.episode_length_buf, self.policy_steps_per_reference
        )
        # Dataset-only asynchronous resets can put environments on different
        # reference-clock phases. The scalar is therefore the fraction of envs at
        # a native boundary; evaluation uses nonzero values to collect this step.
        native_metric_sample = native_metric_mask.float().mean()
        self.extras["native_reference_sample"] = native_metric_sample
        if bool(native_metric_mask.any()):
            native_pos_error = pos_error_norm[native_metric_mask]
            native_rot_error = rot_error_norm[native_metric_mask]
            native_successes = successes[native_metric_mask]
            self._update_adaptive_phase_statistics(
                successes.detach(), native_metric_mask
            )
            self.extras["curr_successes"] = native_successes.float().mean()
            self.extras["curr_tracking_pos_error"] = native_pos_error.mean()
            self.extras["curr_tracking_rot_error"] = native_rot_error.mean()
            # Start a fresh window once the previous one filled a full native
            # reference episode. Only native-boundary samples enter the window.
            if self._track_err_count >= self._track_err_window:
                self._track_err_pos_sum.zero_()
                self._track_err_rot_sum.zero_()
                self._track_err_count = 0
                for buf in (
                    self._track_free_pos_sum,
                    self._track_free_rot_sum,
                    self._track_free_succ_sum,
                    self._track_free_count,
                    self._track_contact_pos_sum,
                    self._track_contact_rot_sum,
                    self._track_contact_succ_sum,
                    self._track_contact_count,
                ):
                    buf.zero_()
            self._track_err_pos_sum += native_pos_error.mean()
            self._track_err_rot_sum += native_rot_error.mean()
            self._track_err_count += 1
            self.extras["tracking_pos_error"] = (
                self._track_err_pos_sum / self._track_err_count
            )
            self.extras["tracking_rot_error"] = (
                self._track_err_rot_sum / self._track_err_count
            )
            if self.enable_force:
                self._accumulate_contact_split_metrics(
                    pos_error_norm,
                    rot_error_norm,
                    successes,
                    env_mask=native_metric_mask,
                )
            self._accumulate_perstep_error(
                pos_error_norm.detach(),
                rot_error_norm.detach(),
                env_mask=native_metric_mask,
            )
            self._accumulate_gain_stats(env_mask=native_metric_mask)
        self._vis_pos_error_norm = pos_error_norm.detach()
        self._vis_rot_error_norm = rot_error_norm.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._dataset_async_reset_enabled:
            reference_time_out = dataset_reference_end_timeouts(
                self.episode_length_buf,
                self._ds_lengths[self.traj_ds_idx],
                self.traj_ds_start,
                chunk_length=self.cfg.tracking.dataset_chunk_length,
                policy_steps_per_reference=self.policy_steps_per_reference,
            )
            time_out = torch.logical_or(time_out, reference_time_out)
        return torch.zeros_like(time_out), time_out

    def _update_adaptive_phase_statistics(
        self,
        successes: torch.Tensor,
        native_metric_mask: torch.Tensor | None = None,
    ):
        """Accumulate strict tracking failures in the active reference bins."""
        if not self._adaptive_phase_enabled:
            return
        if native_metric_mask is None:
            native_metric_mask = torch.ones_like(successes, dtype=torch.bool)
        if not bool(native_metric_mask.any()):
            return
        active_successes = successes[native_metric_mask]
        if self.cfg.tracking.mode == "dataset":
            sequence_ids = self.traj_ds_idx[native_metric_mask]
            reference_steps = self.traj_ds_start[native_metric_mask] + (
                self._policy_step_to_reference_index(
                    self.episode_length_buf[native_metric_mask]
                )
            )
            reference_steps = torch.minimum(
                reference_steps, self._ds_lengths[sequence_ids] - 1
            )
            bin_ids = variable_reference_bin_indices(
                sequence_ids,
                reference_steps,
                bin_offsets=self._adaptive_dataset_bin_offsets,
                bins_per_sequence=self._adaptive_dataset_bins_per_sequence,
                bin_size_steps=self._adaptive_dataset_bin_size_steps,
            )
            # SONIC normalizes occupancy by the number of frames in each bin.
            # A strict tracking miss is still counted at its actual local frame.
            bin_lengths = (
                self._adaptive_dataset_bin_ends[bin_ids]
                - self._adaptive_dataset_bin_starts[bin_ids]
            )
            visits = bin_lengths.to(self._adaptive_phase_visit_counts.dtype).reciprocal()
        else:
            phase_time = self.traj_start_time[native_metric_mask].squeeze(-1) + (
                self.episode_length_buf[native_metric_mask].float()
                * self.step_dt
            )
            bin_ids = phase_bin_indices(
                phase_time,
                bin_size_s=self.cfg.tracking.adaptive_phase_bin_size_s,
                num_bins=self._adaptive_phase_num_bins,
            )
            visits = torch.ones_like(phase_time)
        self._adaptive_phase_visit_counts.index_add_(0, bin_ids, visits)
        self._adaptive_phase_failure_counts.index_add_(
            0,
            bin_ids,
            (~active_successes).to(self._adaptive_phase_failure_counts.dtype),
        )

    def _adaptive_phase_probabilities(
        self, *, start_time_lo: float, start_time_hi: float
    ) -> torch.Tensor:
        return adaptive_phase_sampling_probabilities(
            self._adaptive_phase_failure_counts,
            self._adaptive_phase_visit_counts,
            bin_size_s=self.cfg.tracking.adaptive_phase_bin_size_s,
            start_time_lo=start_time_lo,
            start_time_hi=start_time_hi,
            reference_duration=self.cfg.tracking.circle_reference_duration_s,
            episode_duration=self.max_episode_length_s,
            uniform_rate=self.cfg.tracking.adaptive_phase_uniform_rate,
            failure_rate_max_over_mean=(
                self.cfg.tracking.adaptive_phase_failure_rate_max_over_mean
            ),
            max_probability_multiplier=(
                self.cfg.tracking.adaptive_phase_max_probability_multiplier
            ),
            valid_step_weighting=(
                self.cfg.tracking.adaptive_phase_valid_step_weighting
            ),
        )

    def _adaptive_dataset_probabilities(self) -> torch.Tensor:
        return adaptive_bin_sampling_probabilities(
            self._adaptive_phase_failure_counts,
            self._adaptive_phase_visit_counts,
            self._adaptive_dataset_bin_weights,
            uniform_rate=self.cfg.tracking.adaptive_phase_uniform_rate,
            failure_rate_max_over_mean=(
                self.cfg.tracking.adaptive_phase_failure_rate_max_over_mean
            ),
            max_probability_multiplier=(
                self.cfg.tracking.adaptive_phase_max_probability_multiplier
            ),
        )

    @contextmanager
    def common_eval_context(self):
        """Use the clean controller and enable expensive analytics for eval."""
        original_reset_pos_noise = self.cfg.init.reset_joint_pos_noise
        original_reset_vel_noise = self.cfg.init.reset_joint_vel_noise
        original_reset_noise_prob = self.cfg.init.reset_noise_prob
        original_gain_noise = self.cfg.ctrl.task_prop_gains_noise_level
        original_gain_scale_range = (
            self.cfg.ctrl.task_prop_gains_randomization_scale_range
        )
        original_damping_ratio_range = (
            self.cfg.ctrl.task_deriv_gains_damping_ratio_range
        )
        original_collect_analytics = self._collect_ee_derivative_analytics

        self.cfg.init.reset_joint_pos_noise = 0.0
        self.cfg.init.reset_joint_vel_noise = 0.0
        self.cfg.init.reset_noise_prob = 0.0
        self.cfg.ctrl.task_prop_gains_noise_level = [0.0] * len(
            self.cfg.ctrl.default_task_prop_gains
        )
        self.cfg.ctrl.task_prop_gains_randomization_scale_range = [1.0, 1.0]
        # The aligned real controller uses Kp=400 and Kd=30. Since
        # get_deriv_gains(400) is 40, zeta=0.75 restores the nominal Kd.
        self.cfg.ctrl.task_deriv_gains_damping_ratio_range = [0.75, 0.75]
        self._collect_ee_derivative_analytics = True
        for value in self._episode_ee_derivative_diagnostics.values():
            value.zero_()
        try:
            yield
        finally:
            self.cfg.init.reset_joint_pos_noise = original_reset_pos_noise
            self.cfg.init.reset_joint_vel_noise = original_reset_vel_noise
            self.cfg.init.reset_noise_prob = original_reset_noise_prob
            self.cfg.ctrl.task_prop_gains_noise_level = original_gain_noise
            self.cfg.ctrl.task_prop_gains_randomization_scale_range = (
                original_gain_scale_range
            )
            self.cfg.ctrl.task_deriv_gains_damping_ratio_range = (
                original_damping_ratio_range
            )
            self._collect_ee_derivative_analytics = original_collect_analytics
            analytics_keys = (
                "ee_linear_",
                "ee_angular_",
                "ee_residual_",
                "pose_action_curvature_cost_",
                "gain_action_curvature_cost_",
            )
            for key in list(self.extras):
                if key.startswith(analytics_keys):
                    self.extras.pop(key, None)

    @contextmanager
    def deterministic_eval_context(
        self, *, episode_steps: int, circle_phase_zero: bool = False
    ):
        """Temporarily expose a longer continuous episode for deterministic eval."""
        episode_steps = int(episode_steps)
        if episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        required_reference_steps = math.ceil(
            episode_steps / self.policy_steps_per_reference
        )
        if required_reference_steps > self._reference_buffer_steps:
            raise ValueError(
                "deterministic eval requires "
                f"{required_reference_steps} reference steps, but the buffer has "
                f"{self._reference_buffer_steps}; increase "
                "tracking.reference_buffer_duration_s"
            )

        original_episode_length_s = self.cfg.episode_length_s
        original_phase_range = list(self.cfg.tracking.circle_start_time_range)
        # DirectRLEnv declares timeout at max_episode_length - 1. Keep one
        # internal guard step so the evaluator can execute exactly episode_steps
        # continuous actions before the environment resets.
        self.cfg.episode_length_s = (episode_steps + 1) * self.step_dt
        if circle_phase_zero:
            self.cfg.tracking.circle_start_time_range = [0.0, 0.0]
        try:
            yield
        finally:
            self.cfg.episode_length_s = original_episode_length_s
            self.cfg.tracking.circle_start_time_range = original_phase_range

    def _reset_idx(self, env_ids: torch.Tensor | None):
        import time

        reset_start = time.perf_counter()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._log_reset_metrics(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        partial_dataset_reset = (
            self._dataset_async_reset_enabled and len(env_ids) < self.num_envs
        )
        if not partial_dataset_reset:
            # Analytic resets need the nominal EE pose as their sampling center.
            # The initial all-env dataset reset also takes this path to initialize
            # the shared kinematic-frame cache.
            self._set_franka_to_reset_pose(env_ids)
            self.step_sim_no_action()
            self.nominal_start_pos[env_ids] = self.fingertip_midpoint_pos[
                env_ids
            ].clone()
            self.nominal_start_quat[env_ids] = self.fingertip_midpoint_quat[
                env_ids
            ].clone()
            self._cache_kinematic_frames()

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.reset_actuator_shaping(env_ids)
        # Seed prev velocity with the current (post-reset) velocity so the first
        # step after reset sees zero velocity change instead of a spurious spike.
        self.prev_fingertip_midpoint_linvel[env_ids] = self.fingertip_midpoint_linvel[env_ids]
        self.prev_fingertip_midpoint_angvel[env_ids] = self.fingertip_midpoint_angvel[env_ids]
        self.prev_fingertip_midpoint_linaccel[env_ids] = 0.0
        self.prev_fingertip_midpoint_angaccel[env_ids] = 0.0
        self.ee_jerk_initialized[env_ids] = False
        self._randomize_dynamics(env_ids)
        self._resample_action_latency(env_ids)
        if self._target_update_delay_sampling_mode == "per_episode":
            self._resample_target_update_delay(env_ids)
        self._randomize_controller(env_ids)
        sample_start = time.perf_counter()
        sample_stats = self._sample_reachable_start_and_trajectory(env_ids)
        self.reset_target_schedule(env_ids)
        sample_time = time.perf_counter() - sample_start
        # The sampled reset may carry a nonzero recorded velocity. Seed the
        # smoothness reference after that reset so the first policy step does not
        # compare it against the unrelated generic reset posture.
        self.prev_fingertip_midpoint_linvel[env_ids] = self.fingertip_midpoint_linvel[env_ids]
        self.prev_fingertip_midpoint_angvel[env_ids] = self.fingertip_midpoint_angvel[env_ids]
        if self.enable_force:
            self._sample_force_trajectory(env_ids)
            if self.virtual_contact_enabled:
                self._sample_virtual_contact_surface(env_ids)
        self._update_command()

        # Seed proprio history without retaining stale frames from the prior
        # episode. Dataset chunks use their real lead-in; other modes repeat the
        # current reset frame. `_get_observations` then appends the current frame.
        if self.obs_history_length > 1:
            reset_perturbed_mask = sample_stats["reset_perturbed_mask"]
            if self._uses_recorded_dataset_reset():
                clean_env_ids = env_ids[~reset_perturbed_mask]
                perturbed_env_ids = env_ids[reset_perturbed_mask]
                if clean_env_ids.numel():
                    self._seed_dataset_proprio_history(clean_env_ids)
                if perturbed_env_ids.numel():
                    proprio, _, _, _ = self._compute_obs_parts()
                    self.proprio_history[perturbed_env_ids] = proprio[
                        perturbed_env_ids
                    ].unsqueeze(1)
            else:
                proprio, _, _, _ = self._compute_obs_parts()
                self.proprio_history[env_ids] = proprio[env_ids].unsqueeze(1)

        # Wall-clock reset cost; analytic trajectories are dominated by cuRobo,
        # while recorded dataset chunks bypass the IK worker.
        reset_time = time.perf_counter() - reset_start
        self.extras["log"]["Timing/reset_time_s"] = torch.tensor(reset_time, device=self.device)
        self.extras["log"]["Timing/reachability_sample_time_s"] = torch.tensor(
            sample_time, device=self.device
        )
        avg_jump_str = "[" + ", ".join(f"{v:.3f}" for v in sample_stats["avg_joint_jump"]) + "]"
        if sample_stats["filled"] < sample_stats["total"]:
            print(
                f"[FrankaRobustTrack] reset {len(env_ids)} envs in {reset_time:.3f}s "
                f"(reachability sample {sample_time:.3f}s) | "
                f"candidates sampled {sample_stats['sampled']}, good {sample_stats['good']} "
                f"(unreachable {sample_stats['unreachable']}, discontinuous {sample_stats['discontinuous']}, "
                f"singular {sample_stats['singular']}), "
                f"envs filled {sample_stats['filled']}/{sample_stats['total']} "
                f"in {sample_stats['attempts']} attempt(s) | avg joint jump {avg_jump_str}"
            )

    def _set_franka_to_reset_pose(self, env_ids: torch.Tensor):
        """Teleport the arm to the nominal `ctrl.reset_joints` configuration."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, 0:7] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device).unsqueeze(0)
        if joint_pos.shape[1] > 7:
            joint_pos[:, 7:] = 0.04
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)

    def step_sim_no_action(self):
        """Advance the sim one physics step without applying a policy action.

        Used only during resets, where all environments are stepped together while
        joint states are written directly (teleported) for the IK solve.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _cache_kinematic_frames(self):
        """Cache the constant selected-tool<->hand and world->base transforms once.

        cuRobo IK targets `panda_hand` in the `panda_link0` base frame, while the
        env's reference poses are defined at `tool_frame`. Both directions of the
        tool/hand offset and the base pose are fixed, so they are captured a single
        time after the first reset-pose step. tool->hand converts IK targets; the
        inverse offset shifts cuRobo's hand Jacobian to the controller's tool origin.
        """
        if self._kin_frames_cached:
            return
        tool_pos_w = self.fingertip_midpoint_pos + self.scene.env_origins
        tool_quat_w = self.fingertip_midpoint_quat
        hand_pos_w = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self.hand_body_idx]
        self.hand_in_tool_pos[:], self.hand_in_tool_quat[:] = math_utils.subtract_frame_transforms(
            tool_pos_w, tool_quat_w, hand_pos_w, hand_quat_w
        )
        self.tool_in_hand_pos[:], self.tool_in_hand_quat[:] = math_utils.subtract_frame_transforms(
            hand_pos_w, hand_quat_w, tool_pos_w, tool_quat_w
        )
        self.base_pos_w[:] = self._robot.data.body_pos_w[:, self.base_body_idx]
        self.base_quat_w[:] = self._robot.data.body_quat_w[:, self.base_body_idx]
        self._kin_frames_cached = True

    def _ensure_ik_worker(self):
        """Launch (once) the persistent cuRobo IK subprocess.

        cuRobo IK is run out-of-process because its warp kernels clash with Isaac
        Sim's GPU pipeline in-process (illegal memory access). The worker is a
        clean interpreter with standalone warp; it builds the solver lazily, so the
        first solve request blocks until the solver is ready.
        """
        if self._ik_proc is not None:
            return
        if not self._kin_frames_cached:
            raise RuntimeError("Tool/hand transforms must be cached before launching the cuRobo IK worker.")

        import json
        import os
        import subprocess
        import sys

        from force_tool.utils import curobo_ik_worker

        cfg = self.cfg.init
        # cuRobo/warp default to cuda:0 for the solver and kernel launches, so pin
        # the worker to our physics GPU via CUDA_VISIBLE_DEVICES and address it as
        # cuda:0 inside the worker. Preserve an existing mask: launch wrappers use
        # it to map logical cuda:0 to the assigned physical GPU. Overwriting that
        # mask with self.device.index (which is 0 inside the container) sends every
        # worker to physical GPU 0 on a multi-GPU node.
        device = torch.device(self.device)
        gpu_index = device.index if device.index is not None else 0
        worker_env = os.environ.copy()
        if not worker_env.get("CUDA_VISIBLE_DEVICES", "").strip():
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # The worker shares the GPU with the training/sim process, whose footprint
        # grows over training. expandable_segments lets the worker's allocator
        # return freed memory and fragment less, reducing collisions that OOM'd
        # mid-solve; the worker also adaptively shrinks its batch on OOM.
        alloc_conf = worker_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in alloc_conf:
            worker_env["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{alloc_conf},expandable_segments:True".lstrip(",")
            )
        worker_cfg = {
            "robot_cfg": cfg.ik_robot_cfg,
            "num_seeds": cfg.ik_num_seeds,
            "max_batch_size": self.num_envs * max(cfg.reach_check_waypoints, cfg.reach_oversample),
            "ik_batch_size": cfg.ik_batch_size,
            "position_tolerance": cfg.reach_pos_tol,
            "orientation_tolerance": cfg.reach_rot_tol,
            "use_cuda_graph": cfg.ik_use_cuda_graph,
            "device": "cuda:0",
            # When present, the worker inserts this fixed child into cuRobo's
            # kinematic tree and makes it the actual IK optimization frame.
            "extra_tool_frame": (
                {
                    "name": self.tool_frame_name,
                    "parent_link_name": self.tool_body_name,
                    "fixed_transform": (
                        self.tool_pos_in_body[0].detach().cpu().tolist()
                        + self.tool_quat_in_body[0].detach().cpu().tolist()
                    ),
                }
                if self.curobo_uses_selected_tool_frame
                else None
            ),
            # For legacy body frames cuRobo still solves at panda_hand, so shift
            # its metric Jacobian to the selected tool origin. A directly modeled
            # extra tool already has the correct Jacobian and needs zero shift.
            "metric_tool_offset_pos": (
                [0.0, 0.0, 0.0]
                if self.curobo_uses_selected_tool_frame
                else self.tool_in_hand_pos[0].detach().cpu().tolist()
            ),
        }
        self._ik_worker_mod = curobo_ik_worker
        self._ik_proc = subprocess.Popen(
            [sys.executable, curobo_ik_worker.__file__, json.dumps(worker_cfg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=worker_env,
        )

    def _solve_curobo_ik(
        self,
        tool_pos: torch.Tensor,
        tool_quat: torch.Tensor,
        env_ids: torch.Tensor,
        seed_arm_q: torch.Tensor | None = None,
    ):
        """Solve cuRobo IK (in the worker) for selected-tool targets.

        `tool_pos`/`tool_quat` are (n, W, ...) env-local targets for the envs in
        `env_ids`. A directly modeled virtual tool is sent to cuRobo unchanged in
        the base frame; legacy body-frame targets are converted tool->hand first.
        The targets are sent to the worker as a flat batch of n*W problems, and
        the returned per-target success and arm joints are reshaped back.
        `seed_arm_q`, if given, is a (n, W, 7) tensor of initial joint seeds
        (panda_joint1..7 order)
        passed to the solver so the returned solution stays near it. Returns
        (success (n, W) bool, arm_q (n, W, 7), manip (n, W), cond (n, W)) where
        `manip`/`cond` are the manipulability and Jacobian condition number of each
        solved config (from the geometric Jacobian at the selected tool), used to
        filter near-singular samples.
        """
        self._ensure_ik_worker()
        n_envs, num_wp = tool_pos.shape[0], tool_pos.shape[1]

        env_origins = self.scene.env_origins[env_ids].unsqueeze(1)  # (n, 1, 3)
        base_pos = self.base_pos_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        base_quat = self.base_quat_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_pos = self.hand_in_tool_pos[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_quat = self.hand_in_tool_quat[env_ids].unsqueeze(1).expand(-1, num_wp, -1)

        # Tool target: env-local -> world (env frame is a pure translation).
        tool_pos_w = (tool_pos + env_origins).reshape(-1, 3)
        tool_quat_w = tool_quat.reshape(-1, 4)
        if self.curobo_uses_selected_tool_frame:
            goal_pos_w, goal_quat_w = tool_pos_w, tool_quat_w
        else:
            # Selected tool -> panda_hand for legacy cuRobo robot configs.
            goal_pos_w, goal_quat_w = math_utils.combine_frame_transforms(
                tool_pos_w, tool_quat_w, off_pos.reshape(-1, 3), off_quat.reshape(-1, 4)
            )
        goal_pos_b, goal_quat_b = math_utils.subtract_frame_transforms(
            base_pos.reshape(-1, 3), base_quat.reshape(-1, 4), goal_pos_w, goal_quat_w
        )

        seed_np = None if seed_arm_q is None else seed_arm_q.reshape(-1, 7).detach().cpu().numpy()
        self._ik_worker_mod.send_msg(
            self._ik_proc.stdin,
            (goal_pos_b.detach().cpu().numpy(), goal_quat_b.detach().cpu().numpy(), seed_np),
        )
        response = self._ik_worker_mod.recv_msg(self._ik_proc.stdout)
        if response is None:
            raise RuntimeError(
                f"cuRobo IK worker exited unexpectedly (return code {self._ik_proc.poll()}); see its stderr above."
            )
        success_np, arm_q_np, manip_np, cond_np = response

        success = torch.as_tensor(success_np, device=self.device).view(n_envs, num_wp)
        arm_q = torch.as_tensor(arm_q_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp, 7)
        manip = torch.as_tensor(manip_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp)
        cond = torch.as_tensor(cond_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp)
        return success, arm_q, manip, cond

    def close(self):
        """Tear down the cuRobo IK worker before closing the sim."""
        import subprocess

        if self._ik_proc is not None:
            if self._ik_proc.poll() is None:
                self._ik_proc.stdin.close()
                try:
                    self._ik_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._ik_proc.kill()
            self._ik_proc = None
        super().close()

    def _uses_presampled_circles(self) -> bool:
        """Whether circle resets draw from an offline presampled trajectory set."""
        return (
            self.cfg.tracking.mode == "circle"
            and bool(str(getattr(self.cfg.tracking, "circle_presampled_path", "")).strip())
        )

    _PRESAMPLED_CIRCLE_FORMAT = "robust_track_circle_presampled_v2"

    def _load_presampled_circles(self):
        """Load the offline circle set produced by the presampling script.

        The stored trajectories were accepted under a specific episode length,
        reference clip and waypoint count: their IK chains prove reachability only
        over exactly that horizon, so a mismatch here would silently train on
        trajectories nobody verified for this run's episode. The stored parameters
        are validated against the live config and then held on device in the same
        layout `_assign_params_to_envs` expects, so a reset is a gather, not a
        sampling loop plus a cuRobo round trip.
        """
        import os

        path = os.path.expanduser(str(self.cfg.tracking.circle_presampled_path).strip())
        if os.path.isdir(path):
            split = str(getattr(self.cfg.tracking, "circle_presampled_split", "train")).strip()
            path = os.path.join(path, f"{split or 'train'}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Presampled circle dataset not found: {path}")

        data = np.load(path, allow_pickle=False)
        fmt = str(data["dataset_format"])
        if fmt != self._PRESAMPLED_CIRCLE_FORMAT:
            raise ValueError(
                f"{path} has dataset_format={fmt!r}, expected "
                f"{self._PRESAMPLED_CIRCLE_FORMAT!r}"
            )

        stored_episode_s = float(data["episode_length_s"])
        if not math.isclose(stored_episode_s, self.max_episode_length_s, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"{path} was validated over episode_length_s={stored_episode_s}, but this "
                f"run uses {self.max_episode_length_s}. Regenerate the set with the matching "
                "--episode-length-s, or the trajectories' reachability check does not cover "
                "the episode the policy actually runs."
            )
        stored_reference_s = float(data["circle_reference_duration_s"])
        cfg_reference_s = float(
            getattr(self.cfg.tracking, "circle_reference_duration_s", self.max_episode_length_s)
        )
        if not math.isclose(stored_reference_s, cfg_reference_s, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"{path} was generated with circle_reference_duration_s="
                f"{stored_reference_s}, but this run uses {cfg_reference_s}; the padded "
                "tail of every reference would differ from the one that was IK-checked."
            )
        stored_reference_hz = float(data["reference_hz"])
        if not math.isclose(1.0 / stored_reference_hz, self.reference_dt, rel_tol=1e-6):
            raise ValueError(
                f"{path} solved IK on a {stored_reference_hz} Hz reference grid, but this "
                f"run's reference_dt is {self.reference_dt} s ({1.0 / self.reference_dt:g} Hz). "
                "The stored joint trajectory would not be the one the policy tracks."
            )
        stored_reset_joints = torch.as_tensor(data["reset_joints"], dtype=torch.float32)
        cfg_reset_joints = torch.tensor(self.cfg.ctrl.reset_joints, dtype=torch.float32)
        if not torch.allclose(stored_reset_joints, cfg_reset_joints, atol=1e-6):
            raise ValueError(
                f"{path} was generated around reset_joints={stored_reset_joints.tolist()}, "
                f"but this run uses {cfg_reset_joints.tolist()}; the start-pose anchor "
                "orientation and the null-space posture would both be wrong."
            )
        if self._adaptive_phase_enabled:
            raise ValueError(
                "tracking.circle_start_time_sampling='adaptive' needs to re-solve IK for "
                "each newly drawn phase, so it cannot run against a presampled set whose "
                "phases were fixed at generation time."
            )

        params = {
            key: torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            for key in (
                "start_pos", "start_quat", "u", "v", "radius", "omega",
                "start_time", "rot_axis", "rot_speed", "rot_angle",
            )
        }
        # Line-mode params are inactive for circles but `_assign_params_to_envs`
        # copies every buffer uniformly, so give them the same harmless zeros the
        # online circle sampler writes.
        num_trajs = params["start_pos"].shape[0]
        for key, dim in (("dir", 3), ("speed", 1), ("length", 1)):
            params[key] = torch.zeros((num_trajs, dim), device=self.device)

        # Only the start configuration is needed at reset: the reference itself is
        # regenerated analytically from the circle params, exactly as before. The
        # file's full `joint_pos` chain (N, K, S, 7) is the offline validation
        # record and is deliberately left unread -- `np.load` on an npz is lazy, so
        # not touching the key keeps a few hundred MB off the GPU.
        start_joint_pos = torch.as_tensor(
            data["start_joint_pos"], dtype=torch.float32, device=self.device
        )  # (N, K, 7)
        branch_valid = torch.as_tensor(
            data["branch_valid"], dtype=torch.bool, device=self.device
        )  # (N, K)
        if not bool(branch_valid.any(dim=1).all()):
            raise ValueError(f"{path} contains trajectories with no valid start branch")

        self._presampled_circle = {
            "path": path,
            "params": params,
            # Checked once against Isaac's own FK at the first reset; see
            # `_assign_presampled_circles`.
            "nominal_tool_quat": torch.as_tensor(
                data["nominal_tool_quat_wxyz"], dtype=torch.float32, device=self.device
            ),
            "anchor_checked": False,
            "start_joint_pos": start_joint_pos,
            # Branch weights for `torch.multinomial`; every trajectory has at least
            # one valid branch, checked above.
            "branch_weights": branch_valid.float(),
            "num_trajs": num_trajs,
        }
        print(
            f"[presampled circles] {num_trajs} trajectories from {path} | "
            f"{start_joint_pos.shape[1]} start branches/traj, "
            f"{branch_valid.float().sum(dim=1).mean().item():.2f} valid on average"
        )

    def _assign_presampled_circles(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Draw a presampled trajectory and one of its start branches per env.

        A trajectory is drawn uniformly and *then* one of its valid start branches,
        so the Cartesian reference distribution is exactly the presampled one while
        the arm posture still varies across resets. Sampling uniformly over
        (trajectory, branch) pairs instead would over-represent the trajectories
        that happen to admit more IK branches.

        Returns the (len(env_ids), 7) start arm configuration to reset into.
        """
        store = self._presampled_circle
        if not store["anchor_checked"]:
            # The generator anchors every start orientation on cuRobo's FK of
            # `panda_hand` plus the fixed hand->fingertip offset; the env derives
            # the same pose from Isaac's FK of the real tool body. They describe
            # one physical TCP, but only if the run's `tool_frame` is the one the
            # set was generated for -- so compare them once, where the env's own
            # nominal pose is finally available.
            stored = store["nominal_tool_quat"]
            live = self.nominal_start_quat[env_ids[0]]
            # Quaternion double cover: q and -q are the same rotation.
            if float((stored * live).sum().abs()) < 1.0 - 1.0e-3:
                raise ValueError(
                    f"{store['path']} was generated around tool orientation "
                    f"{stored.tolist()}, but this run's FK at reset_joints gives "
                    f"{live.tolist()}. The presampled circles are anchored on a "
                    "different tool frame than tool_frame="
                    f"{getattr(self, 'tool_frame_name', '?')!r} resolves to."
                )
            store["anchor_checked"] = True
        traj_idx = torch.randint(
            store["num_trajs"], (len(env_ids),), device=self.device
        )
        branch_idx = torch.multinomial(store["branch_weights"][traj_idx], 1).squeeze(-1)
        self._assign_params_to_envs(env_ids, store["params"], traj_idx)
        return store["start_joint_pos"][traj_idx, branch_idx]

    def _sample_reachable_start_and_trajectory(self, env_ids: torch.Tensor):
        """Sample a start pose + trajectory per env, keeping only continuously reachable ones.

        Because the reachability + continuity check rejects many analytic samples, each env
        is *oversampled*: `init.reach_oversample` independent candidate trajectories
        are drawn per pending env and evaluated together. For every candidate we
        walk the `reach_check_waypoints` waypoints spanning the moving trajectory
        from first to last, solving cuRobo IK one waypoint at a time and seeding
        from the previous solution. Dataset chunks bypass cuRobo entirely: their
        recorded joints are already the known joint solution for every waypoint.
        An analytic candidate is accepted only if every
        waypoint is
        reachable (IK success within `reach_pos_tol`/`reach_rot_tol`) *and*
        consecutive joint solutions never jump by more than
        `reach_joint_diff_threshold` on any joint. The first accepted candidate is
        kept per env; envs with no accepted candidate are re-oversampled (up to
        `init.max_reach_attempts`). Finally every reset env is placed at its accepted
        cuRobo start solution, or at the exact recorded q[t] for a dataset chunk.
        """
        cfg = self.cfg.init
        # Reset-path diagnostics, shared by the online sampler and the presampled
        # set so the logged counters mean the same thing either way.
        total_sampled = total_good = 0
        total_reach_fail = total_cont_fail = total_sing_fail = 0
        total_containment_fail = 0
        attempt = 0
        # Per-joint |Δq| accumulated over waypoint transitions where both endpoints
        # were reachable, to report the average continuity of the sampled chains.
        joint_jump_sum = torch.zeros(7, device=self.device)
        joint_jump_count = 0
        # cuRobo arm-joint solution for each env's start pose; filled during the check.
        start_arm_q = self.joint_pos[:, 0:7].clone()
        bad_envs = env_ids[:0]

        if self._presampled_circle is not None:
            # Offline-validated circles: every gate below (reachability, waypoint
            # continuity, Jacobian conditioning, workspace containment) was already
            # applied at generation time under this run's `init.*` profile, and each
            # trajectory carries its own solved start branches. A reset is therefore
            # a draw plus a gather -- no candidate loop and no cuRobo round trip.
            start_arm_q[env_ids] = self._assign_presampled_circles(env_ids)
            total_sampled = total_good = int(len(env_ids))
        else:
            bad_envs = env_ids.clone()
            num_wp = cfg.reach_check_waypoints
            recorded_dataset_reset = self._uses_recorded_dataset_reset()
            # A recorded dataset sample is already a reachable joint-space trajectory;
            # there is no reason to oversample it or reconstruct it with IK.
            k = 1 if recorded_dataset_reset else max(1, int(cfg.reach_oversample))
            # Singularity filter (getattr keeps old restored env.pkl configs working:
            # absent -> disabled). A waypoint config is rejected when its geometric-
            # Jacobian manipulability is too low or its condition number too high.
            sing_check = bool(getattr(cfg, "singularity_check", False))
            min_manip = float(getattr(cfg, "min_manipulability", 0.0))
            max_cond = float(getattr(cfg, "max_jac_cond", float("inf")))
            containment_enabled = bool(getattr(cfg, "enable_ee_containment", False))
            if containment_enabled:
                containment_box = torch.as_tensor(
                    cfg.ee_containment_box, device=self.device, dtype=torch.float32
                )
                if containment_box.shape != (3, 2):
                    raise ValueError("init.ee_containment_box must have shape (3, 2)")
            reach_duration = self.max_episode_length_s
            if self.cfg.tracking.mode == "dataset" and str(
                getattr(self.cfg.tracking, "dataset_chunk_start_mode", "episode_start")
            ) == "random":
                chunk_length = int(getattr(self.cfg.tracking, "dataset_chunk_length", 0))
                if chunk_length > 0:
                    reach_duration = max(chunk_length - 1, 0) * self.reference_dt
            waypoint_times = torch.linspace(0.0, reach_duration, num_wp, device=self.device)

            # Accepted (reachable + continuous) candidates accumulated across attempts.
            # Any envs still pending after the last attempt are backfilled by resampling
            # (with replacement) from these known-good trajectories rather than arbitrary
            # ones; per-env domain randomization still differentiates repeated trajectories.
            accepted_store = {key: [] for key in self._traj_param_buffers()}
            accepted_start_q = []
            # Optionally seed the first waypoint's IK with reset_joints so the start
            # config lands on that arm branch (later waypoints seed from the previous
            # solution and follow it); mirrors factory/forge's teleport-to-reset_joints.
            reset_seed = None
            if getattr(cfg, "seed_ik_with_reset_joints", False):
                reset_seed = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)
            while True:
                n_bad = len(bad_envs)
                total_sampled += n_bad * k
                # Draw a pool of n_bad*k candidate trajectories. `cand_env_ids` only
                # feeds the IK frame conversion, which is identical across envs, so the
                # candidates are not actually bound to specific envs.
                params = self._sample_candidate_params(bad_envs, k)
                cand_env_ids = bad_envs.repeat_interleave(k)
                containment_ok = torch.ones(
                    n_bad * k, dtype=torch.bool, device=self.device
                )
                if containment_enabled:
                    num_containment_checks = max(
                        2, int(math.ceil(reach_duration / self.reference_dt)) + 1
                    )
                    containment_times = torch.linspace(
                        0.0,
                        reach_duration,
                        num_containment_checks,
                        device=self.device,
                    )
                    for check_time in containment_times:
                        elapsed_time = check_time.expand(n_bad * k, 1)
                        pos_i, _ = self._eval_pose_from_params(params, elapsed_time)
                        containment_ok &= axis_aligned_containment_mask(
                            pos_i, containment_box
                        )
                recorded_start_q = None
                if recorded_dataset_reset:
                    recorded_start_q = self._ds_joint_pos[
                        params["ds_idx"], params["ds_start"]
                    ]

                # Walk the waypoints in order, seeding each solve with the previous
                # solution. Track reachability and continuity separately so we can report
                # which one is rejecting candidates.
                reach_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
                cont_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
                sing_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
                cand_start_q = torch.zeros((n_bad * k, 7), device=self.device)
                prev_q = None
                prev_success = None
                for i in range(num_wp):
                    if i == 0 and recorded_start_q is not None:
                        # This exact q/pose pair was recorded by Forge and was
                        # validated while loading the dataset. Avoid re-solving a
                        # redundant 7-DoF IK target just to rediscover a potentially
                        # different branch; seed the next checked waypoint from q[t].
                        cand_start_q = recorded_start_q.clone()
                        prev_q = recorded_start_q
                        prev_success = torch.ones_like(reach_ok)
                        continue
                    if recorded_start_q is not None:
                        # Use the demonstration's known joint solution at every checked
                        # waypoint. Clamp padded chunks to their final valid sample.
                        relative_idx = int(
                            round(float(waypoint_times[i].item()) / self.reference_dt)
                        )
                        source_idx = torch.minimum(
                            params["ds_start"] + relative_idx,
                            self._ds_lengths[params["ds_idx"]] - 1,
                        )
                        arm_q = self._ds_joint_pos[params["ds_idx"], source_idx]
                        success = torch.ones_like(reach_ok)
                        manip = torch.full_like(reach_ok, float("inf"), dtype=torch.float32)
                        cond = torch.zeros_like(reach_ok, dtype=torch.float32)
                    else:
                        elapsed_time = torch.full((n_bad * k, 1), waypoint_times[i].item(), device=self.device)
                        pos_i, quat_i = self._eval_pose_from_params(params, elapsed_time)
                        if prev_q is not None:
                            seed = prev_q.unsqueeze(1)
                        elif reset_seed is not None:
                            # First waypoint: seed from reset_joints (does not set prev_q, so
                            # the i==0 continuity check stays skipped as before).
                            seed = reset_seed.unsqueeze(0).expand(n_bad * k, 7).unsqueeze(1)
                        else:
                            seed = None
                        success, arm_q, manip, cond = self._solve_curobo_ik(
                            pos_i.unsqueeze(1), quat_i.unsqueeze(1), cand_env_ids, seed_arm_q=seed
                        )
                        success = success[:, 0]
                        arm_q = arm_q[:, 0]
                    reach_ok &= success
                    if sing_check and recorded_start_q is None:
                        # Only gate reachable waypoints (unreachable ones already reject
                        # the candidate and carry meaningless IK solutions).
                        wp_sing_ok = (manip[:, 0] >= min_manip) & (cond[:, 0] <= max_cond)
                        sing_ok &= wp_sing_ok | ~success
                    if prev_q is not None:
                        joint_diff = (arm_q - prev_q).abs()  # (n*k, 7)
                        if recorded_start_q is None:
                            cont_ok &= joint_diff.amax(dim=-1) <= cfg.reach_joint_diff_threshold
                        # Only count transitions where both endpoints were reachable.
                        both_ok = prev_success & success
                        joint_jump_sum += joint_diff[both_ok].sum(dim=0)
                        joint_jump_count += int(both_ok.sum().item())
                    if i == 0:
                        cand_start_q = arm_q.clone()
                    prev_q = arm_q
                    prev_success = success

                accept = reach_ok & cont_ok & sing_ok & containment_ok
                total_good += int(accept.sum().item())
                total_reach_fail += int((~reach_ok).sum().item())
                total_cont_fail += int((reach_ok & ~cont_ok).sum().item())
                total_sing_fail += int((reach_ok & cont_ok & ~sing_ok).sum().item())
                total_containment_fail += int(
                    (reach_ok & cont_ok & sing_ok & ~containment_ok).sum().item()
                )

                # The whole pool of n_bad*k candidates is fungible: the start pose and
                # trajectory are env-independent and the IK is solved in the (shared)
                # base frame, so any accepted candidate is valid for any env. Randomly
                # draw accepted candidates from the pool and assign them to the pending
                # envs rather than tying each env to its own k candidates.
                accepted_idx = accept.nonzero(as_tuple=False).squeeze(-1)
                if accepted_idx.numel() > 0:
                    for key in self._traj_param_buffers():
                        accepted_store[key].append(params[key][accepted_idx])
                    accepted_start_q.append(cand_start_q[accepted_idx])
                    perm = accepted_idx[torch.randperm(accepted_idx.numel(), device=self.device)]
                    take = perm[: len(bad_envs)]
                    fill_envs = bad_envs[: take.numel()]
                    self._assign_params_to_envs(fill_envs, params, take)
                    start_arm_q[fill_envs] = cand_start_q[take]
                    bad_envs = bad_envs[take.numel() :]

                attempt += 1
                if bad_envs.shape[0] == 0 or attempt >= cfg.max_reach_attempts:
                    # Backfill: envs that never got their own accepted candidate reuse a
                    # randomly drawn accepted (reachable + continuous) trajectory. Repeats
                    # are fine since domain randomization still differs per env. If nothing
                    # was ever accepted, fall back to an arbitrary candidate from the pool.
                    if bad_envs.shape[0] > 0:
                        if len(accepted_start_q) > 0:
                            store = {key: torch.cat(vals, dim=0) for key, vals in accepted_store.items()}
                            store_q = torch.cat(accepted_start_q, dim=0)
                            pick = torch.randint(store_q.shape[0], (bad_envs.shape[0],), device=self.device)
                            self._assign_params_to_envs(bad_envs, store, pick)
                            start_arm_q[bad_envs] = store_q[pick]
                        elif containment_enabled:
                            raise RuntimeError(
                                "No trajectory satisfied init.ee_containment_box after "
                                f"{cfg.max_reach_attempts} reachability attempts"
                            )
                        else:
                            fb = torch.arange(bad_envs.shape[0], device=self.device)
                            self._assign_params_to_envs(bad_envs, params, fb)
                            start_arm_q[bad_envs] = cand_start_q[fb]
                    break

        # Discretize the accepted trajectory onto the native reference grid.
        self._build_trajectory_buffer(env_ids)

        # Place every reset env at its reachable start configuration. Dataset
        # chunks replace the cuRobo result with their exact recorded q[t]/qdot[t].
        start_arm_vel = torch.zeros((len(env_ids), 7), device=self.device)
        if self._uses_recorded_dataset_reset():
            ds_idx = self.traj_ds_idx[env_ids]
            ds_start = self.traj_ds_start[env_ids]
            recorded_q = self._ds_joint_pos[ds_idx, ds_start]
            prev_idx = (ds_start - 1).clamp(min=0)
            recorded_prev_q = self._ds_joint_pos[ds_idx, prev_idx]
            start_arm_q[env_ids] = recorded_q
            start_arm_vel = (recorded_q - recorded_prev_q) / self.reference_dt

        # Perturb only the physical reset state, after IK/reachability selection.
        # This keeps the sampled reference trajectory intact while exposing the
        # controller to reset errors.  Soft limits provide the verified PhysX
        # safety margin, so a perturbation can never push an arm joint outside it.
        q_noise = float(getattr(self.cfg.init, "reset_joint_pos_noise", 0.0))
        qd_noise = float(getattr(self.cfg.init, "reset_joint_vel_noise", 0.0))
        reset_noise_prob = float(getattr(self.cfg.init, "reset_noise_prob", 1.0))
        if not 0.0 <= reset_noise_prob <= 1.0:
            raise ValueError(
                f"init.reset_noise_prob must be in [0, 1], got {reset_noise_prob}"
            )
        reset_perturbed_mask = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        if q_noise or qd_noise:
            reset_perturbed_mask = (
                torch.rand(len(env_ids), device=self.device)
                < reset_noise_prob
            )
            reset_noise_gate = reset_perturbed_mask.unsqueeze(-1)
            if q_noise:
                q_delta = (
                    (2.0 * torch.rand_like(start_arm_q[env_ids, 0:7]) - 1.0)
                    * q_noise
                    * reset_noise_gate
                )
                joint_limits = self._robot.data.soft_joint_pos_limits[env_ids, 0:7]
                start_arm_q[env_ids] = torch.minimum(
                    torch.maximum(start_arm_q[env_ids] + q_delta, joint_limits[..., 0]),
                    joint_limits[..., 1],
                )
            if qd_noise:
                start_arm_vel = start_arm_vel + (
                    2.0 * torch.rand_like(start_arm_vel) - 1.0
                ) * qd_noise * reset_noise_gate
        self.extras["log"]["Init/reset_perturbed_fraction"] = (
            reset_perturbed_mask.float().mean()
        )
        self.joint_pos[env_ids, 0:7] = start_arm_q[env_ids]
        # Anchor the OSC nullspace posture. "start" uses each env's own cuRobo start
        # config (per-env); "reset_joints"/"default_dof_pos" use a fixed posture so the
        # redundancy resolution biases toward a consistent config (factory/forge use a
        # fixed anchor -> "default_dof_pos" matches them exactly).
        posture = getattr(self.cfg.ctrl, "nullspace_posture", "start")
        if posture == "reset_joints":
            self.nominal_start_joint_pos[env_ids] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)
        elif posture == "default_dof_pos":
            self.nominal_start_joint_pos[env_ids] = torch.tensor(
                self.cfg.ctrl.default_dof_pos_tensor, device=self.device
            )
        else:
            self.nominal_start_joint_pos[env_ids] = start_arm_q[env_ids]
        if self.joint_pos.shape[1] > 7:
            self.joint_pos[env_ids, 7:] = 0.04
        self.joint_vel[env_ids] = 0.0
        self.joint_vel[env_ids, 0:7] = start_arm_vel
        self.ctrl_target_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._robot.write_joint_state_to_sim(
            self.joint_pos[env_ids], self.joint_vel[env_ids], env_ids=env_ids
        )
        self._robot.set_joint_position_target(
            self.ctrl_target_joint_pos[env_ids], env_ids=env_ids
        )
        self._robot.set_joint_effort_target(
            torch.zeros_like(self.joint_pos[env_ids]), env_ids=env_ids
        )
        reset_joint_pos = self.joint_pos[env_ids].clone()
        reset_joint_vel = self.joint_vel[env_ids].clone()
        reset_perturbed = bool(q_noise or qd_noise) and (
            reset_noise_prob > 0.0
        )
        partial_dataset_reset = (
            self._dataset_async_reset_enabled and len(env_ids) < self.num_envs
        )
        if partial_dataset_reset:
            # The selected dataset joints are already written directly to PhysX.
            # Refresh tensor views without advancing the other environments.
            self.scene.update(dt=0.0)
            self._compute_intermediate_values()
        else:
            self.step_sim_no_action()
        if not partial_dataset_reset and (
            self._uses_recorded_dataset_reset() or reset_perturbed
        ):
            # The refresh step above makes PhysX/Jacobians reflect the sampled
            # configuration, but a nonzero qdot also advances q by one physics
            # step. Restore the exact requested reset state before the first
            # observation.
            self._robot.write_joint_state_to_sim(
                reset_joint_pos, reset_joint_vel, env_ids=env_ids
            )
            self.scene.update(dt=0.0)
            self._compute_intermediate_values()
        num_filled = int(len(env_ids) - bad_envs.shape[0])
        self.extras["log"]["Init/unreachable_envs"] = torch.tensor(float(bad_envs.shape[0]), device=self.device)
        self.extras["log"]["Init/candidates_sampled"] = torch.tensor(float(total_sampled), device=self.device)
        self.extras["log"]["Init/candidates_good"] = torch.tensor(float(total_good), device=self.device)
        self.extras["log"]["Init/candidates_unreachable"] = torch.tensor(float(total_reach_fail), device=self.device)
        self.extras["log"]["Init/candidates_discontinuous"] = torch.tensor(float(total_cont_fail), device=self.device)
        self.extras["log"]["Init/candidates_singular"] = torch.tensor(float(total_sing_fail), device=self.device)
        self.extras["log"]["Init/candidates_outside_ee_containment"] = torch.tensor(
            float(total_containment_fail), device=self.device
        )
        if self.cfg.tracking.mode == "dataset":
            sampled_start = self.traj_ds_start[env_ids].float()
            self.extras["log"]["Init/dataset_chunk_start_mean"] = sampled_start.mean()
            self.extras["log"]["Init/dataset_chunk_start_min"] = sampled_start.min()
            self.extras["log"]["Init/dataset_chunk_start_max"] = sampled_start.max()
            self.extras["log"]["Init/dataset_joint_speed_mean"] = start_arm_vel.norm(dim=-1).mean()
        avg_joint_jump = joint_jump_sum / max(joint_jump_count, 1)
        for j in range(7):
            self.extras["log"][f"Init/avg_joint_jump_{j}"] = avg_joint_jump[j]
        return {
            "sampled": total_sampled,
            "good": total_good,
            "unreachable": total_reach_fail,
            "discontinuous": total_cont_fail,
            "singular": total_sing_fail,
            "filled": num_filled,
            "total": int(len(env_ids)),
            "attempts": attempt,
            "avg_joint_jump": avg_joint_jump.tolist(),
            "reset_perturbed_mask": reset_perturbed_mask,
        }

    @staticmethod
    def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Batched shortest-path spherical interpolation of (w, x, y, z) quats.

        `q0`, `q1` are (m, 4) unit quats; `t` is (m,) in [0, 1]. Falls back to a
        normalized lerp when the two quats are nearly parallel (sin theta ~ 0).
        """
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = dot.abs().clamp(max=1.0)
        theta = torch.acos(dot)  # (m, 1)
        sin_theta = torch.sin(theta)
        small = sin_theta < 1.0e-6
        t = t.unsqueeze(-1)
        w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp(min=1.0e-6))
        w1 = torch.where(small, t, torch.sin(t * theta) / sin_theta.clamp(min=1.0e-6))
        q = w0 * q0 + w1 * q1
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _resample_pose_sequence(self, pos: torch.Tensor, quat: torch.Tensor, length: int):
        """Resample a (n, 3)/(n, 4) pose sequence onto `length` evenly-spaced samples."""
        n = pos.shape[0]
        tgt = torch.linspace(0.0, n - 1, length, device=pos.device)
        i0 = tgt.floor().long().clamp(0, n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        frac = (tgt - i0.float()).unsqueeze(-1)
        p = pos[i0] * (1.0 - frac) + pos[i1] * frac
        q = self._quat_slerp(quat[i0], quat[i1], frac.squeeze(-1))
        return p, q

    @staticmethod
    def _resample_vec_sequence(vec: torch.Tensor, length: int) -> torch.Tensor:
        """Linearly resample a (n, d) sequence onto `length` evenly-spaced samples."""
        n = vec.shape[0]
        tgt = torch.linspace(0.0, n - 1, length, device=vec.device)
        i0 = tgt.floor().long().clamp(0, n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        frac = (tgt - i0.float()).unsqueeze(-1)
        return vec[i0] * (1.0 - frac) + vec[i1] * frac

    def _initialize_dataset_adaptive_sampling(self):
        """Create SONIC-style flattened bins for variable-length dataset clips."""
        bin_size_steps = max(
            1,
            int(
                round(
                    float(self.cfg.tracking.adaptive_phase_bin_size_s)
                    / self.reference_dt
                )
            ),
        )
        valid_step_weighting = bool(
            self.cfg.tracking.adaptive_phase_valid_step_weighting
        ) and not self._dataset_async_reset_enabled
        (
            self._adaptive_dataset_bin_offsets,
            self._adaptive_dataset_bins_per_sequence,
            self._adaptive_dataset_bin_sequence_ids,
            self._adaptive_dataset_bin_starts,
            self._adaptive_dataset_bin_ends,
            self._adaptive_dataset_bin_weights,
        ) = variable_reference_bin_layout(
            self._ds_lengths,
            bin_size_steps=bin_size_steps,
            sequence_length_agnostic=True,
            chunk_length=int(self.cfg.tracking.dataset_chunk_length),
            valid_step_weighting=valid_step_weighting,
        )
        self._adaptive_dataset_bin_size_steps = bin_size_steps
        self._adaptive_phase_num_bins = int(
            self._adaptive_dataset_bin_sequence_ids.numel()
        )
        init_count = float(self.cfg.tracking.adaptive_phase_init_count)
        self._adaptive_phase_visit_counts = torch.full(
            (self._adaptive_phase_num_bins,),
            init_count,
            device=self.device,
        )
        self._adaptive_phase_failure_counts = torch.full_like(
            self._adaptive_phase_visit_counts, init_count
        )

    def _load_dataset_trajectories(self):
        """Load offline demonstration EE trajectories for ``mode='dataset'``.

        Reads the `dataset_pose_key` (N, 7) pose field (selected-tool pos + quat,
        wxyz, env-local frame) from the HDF5 dataset and splits it into episodes via
        `episode_starts`/`episode_ends`. With `dataset_chunk_length > 0`, the raw
        episodes and their recorded arm joints are retained. Reset samples a random
        within-episode chunk start, and runtime lookup repeats the chunk's final pose
        through the unused episode/lookahead tail. A zero chunk length retains the
        legacy even-resampling behavior for old configs.
        """
        import h5py

        tcfg = self.cfg.tracking
        path = tcfg.dataset_path
        if not path:
            raise ValueError("tracking.mode='dataset' requires tracking.dataset_path")
        warp_prob = float(getattr(tcfg, "dataset_warp_prob", 0.0))
        warp_pos_max = float(getattr(tcfg, "dataset_warp_pos_max", 0.0))
        warp_rot_max = float(getattr(tcfg, "dataset_warp_rot_max", 0.0))
        warp_strategy = str(getattr(tcfg, "dataset_warp_strategy", "smooth_bump"))
        shape_fraction = float(getattr(tcfg, "dataset_warp_shape_fraction", 0.0))
        cycles_range = getattr(tcfg, "dataset_warp_cycles_range", [0.5, 1.5])
        valid_warp_strategies = {"smooth_bump", "constant_scale_shape"}
        if warp_strategy not in valid_warp_strategies:
            raise ValueError(
                f"tracking.dataset_warp_strategy must be one of {sorted(valid_warp_strategies)}, "
                f"got {warp_strategy!r}"
            )
        if not 0.0 <= warp_prob <= 1.0:
            raise ValueError(f"tracking.dataset_warp_prob must be in [0, 1], got {warp_prob}")
        if warp_pos_max < 0.0 or warp_rot_max < 0.0:
            raise ValueError("tracking dataset warp magnitudes must be non-negative")
        if not 0.0 <= shape_fraction <= 1.0:
            raise ValueError("tracking.dataset_warp_shape_fraction must be in [0, 1]")
        if len(cycles_range) != 2 or cycles_range[0] < 0.0 or cycles_range[1] < cycles_range[0]:
            raise ValueError("tracking.dataset_warp_cycles_range must be [min, max] with 0 <= min <= max")

        length = self._traj_len
        chunk_length = int(getattr(tcfg, "dataset_chunk_length", 0))
        if chunk_length < 0:
            raise ValueError(
                f"tracking.dataset_chunk_length must be non-negative, got {chunk_length}"
            )
        if chunk_length > length:
            raise ValueError(
                "tracking.dataset_chunk_length cannot exceed the runtime trajectory "
                f"buffer length ({chunk_length} > {length})"
            )
        if chunk_length > 0 and warp_prob > 0.0 and warp_strategy != "smooth_bump":
            raise ValueError(
                "Recorded-joint chunk reset requires an endpoint-preserving smooth_bump "
                "warp (or dataset_warp_prob=0), so the warped chunk still begins at q[t]"
            )
        reference_length = int(getattr(tcfg, "dataset_reference_length", 0))
        if chunk_length > 0:
            reference_length = chunk_length
        elif reference_length <= 0:
            reference_length = length
        if reference_length > length:
            raise ValueError(
                "tracking.dataset_reference_length cannot exceed the runtime "
                f"trajectory buffer length ({reference_length} > {length})"
            )
        want_force = self.enable_force and bool(tcfg.dataset_force_key)
        with h5py.File(path, "r") as f:
            pose_key = str(getattr(tcfg, "dataset_pose_key", "tool_pose"))
            position_key = str(getattr(tcfg, "dataset_position_key", ""))
            axis_angle_key = str(getattr(tcfg, "dataset_axis_angle_key", ""))
            if pose_key:
                pose = torch.as_tensor(f[pose_key][:], dtype=torch.float32)
            elif position_key and axis_angle_key:
                position = torch.as_tensor(f[position_key][:], dtype=torch.float32)
                rotation_vector = torch.as_tensor(
                    f[axis_angle_key][:], dtype=torch.float32
                )
                if position.ndim != 2 or position.shape[1] != 3:
                    raise ValueError(
                        f"Invalid {position_key!r} shape {tuple(position.shape)}; "
                        "expected (N, 3)"
                    )
                if rotation_vector.shape != position.shape:
                    raise ValueError(
                        f"Invalid {axis_angle_key!r} shape {tuple(rotation_vector.shape)}; "
                        f"expected {tuple(position.shape)}"
                    )
                angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
                axis = rotation_vector / angle.clamp_min(1.0e-12).unsqueeze(-1)
                quat = torch_utils.quat_from_angle_axis(angle, axis)
                zero_rotation = angle < 1.0e-12
                if torch.any(zero_rotation):
                    quat[zero_rotation] = quat.new_tensor([1.0, 0.0, 0.0, 0.0])
                pose = torch.cat((position, quat), dim=-1)
            else:
                raise ValueError(
                    "Dataset tracking requires either dataset_pose_key or both "
                    "dataset_position_key and dataset_axis_angle_key"
                )
            starts = f["episode_starts"][:]
            ends = f["episode_ends"][:]
            success = f["trial_success"][:] if "trial_success" in f else None
            force = torch.as_tensor(f[tcfg.dataset_force_key][:], dtype=torch.float32) if want_force else None
            state = None
            use_recorded_reset = chunk_length > 0 and str(
                getattr(tcfg, "dataset_chunk_start_mode", "episode_start")
            ) == "random"
            if use_recorded_reset:
                state_key = str(getattr(tcfg, "dataset_state_key", "low_dim_state"))
                joint_key = str(getattr(tcfg, "dataset_joint_pos_key", ""))
                joint_offset = int(getattr(tcfg, "dataset_joint_pos_offset", 13))
                if joint_key:
                    if joint_key not in f:
                        raise KeyError(
                            f"Dataset chunk reset requires {joint_key!r} with recorded arm joints: {path}"
                        )
                    state = torch.as_tensor(f[joint_key][:], dtype=torch.float32)
                elif state_key not in f:
                    raise KeyError(
                        f"Dataset chunk reset requires {state_key!r} with recorded arm joints: {path}"
                    )
                else:
                    state = torch.as_tensor(f[state_key][:], dtype=torch.float32)
                if state.ndim != 2 or joint_offset < 0 or state.shape[1] < joint_offset + 7:
                    raise ValueError(
                        f"Invalid {(joint_key or state_key)!r} shape {tuple(state.shape)} or joint offset "
                        f"{joint_offset}; expected at least {joint_offset + 7} columns"
                    )
                if state.shape[0] != pose.shape[0]:
                    raise ValueError(
                        f"{joint_key or state_key} has {state.shape[0]} rows but the "
                        f"dataset pose has {pose.shape[0]} rows"
                    )
                if not joint_key and not torch.allclose(
                    state[:, :7], pose, atol=1.0e-5, rtol=1.0e-5
                ):
                    raise ValueError(
                        f"{state_key}[:, :7] does not align with {pose_key}; "
                        "refusing to use its recorded joints"
                    )

        storage_length = max(int(en) - int(s) for s, en in zip(starts, ends))
        pos_list, quat_list, phase_list, wr_list, joint_list, raw_lengths = [], [], [], [], [], []
        torque_norm_samples = []
        for e in range(len(starts)):
            if tcfg.dataset_only_success and success is not None and float(success[e]) < 0.5:
                continue
            s, en = int(starts[e]), int(ends[e])
            if en - s < 2:
                continue
            seg_pos = pose[s:en, :3].to(self.device)
            seg_quat = pose[s:en, 3:7].to(self.device)
            seg_quat = seg_quat / seg_quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            raw_length = seg_pos.shape[0]
            if chunk_length > 0:
                p, q = seg_pos, seg_quat
                storage_pad = storage_length - raw_length
                if storage_pad > 0:
                    p = torch.cat((p, p[-1:].expand(storage_pad, -1)), dim=0)
                    q = torch.cat((q, q[-1:].expand(storage_pad, -1)), dim=0)
                raw_lengths.append(raw_length)
                if state is not None:
                    arm_q = state[s:en, joint_offset : joint_offset + 7].to(self.device)
                    if not torch.isfinite(arm_q).all():
                        raise ValueError(f"Episode {e} contains non-finite recorded arm joints")
                    if storage_pad > 0:
                        arm_q = torch.cat(
                            (arm_q, arm_q[-1:].expand(storage_pad, -1)), dim=0
                        )
                    joint_list.append(arm_q)
            else:
                p, q = self._resample_pose_sequence(
                    seg_pos, seg_quat, reference_length
                )
            if chunk_length == 0 and reference_length < length:
                pad = length - reference_length
                p = torch.cat((p, p[-1:].expand(pad, -1)), dim=0)
                q = torch.cat((q, q[-1:].expand(pad, -1)), dim=0)
            pos_list.append(p)
            quat_list.append(q)
            if chunk_length > 0:
                # Chunk-relative phase is generated after the random start is
                # sampled; keep a harmless placeholder with the storage shape.
                phase = torch.zeros(storage_length, device=self.device)
            else:
                phase = torch.linspace(0.0, 1.0, length, device=self.device)
            phase_list.append(phase)
            if force is not None:
                if force.ndim != 2 or force.shape[1] not in (3, 6):
                    raise ValueError(
                        "Dataset wrench must have exactly 3 force columns or 6 "
                        f"force/torque columns, got {tuple(force.shape)}"
                    )
                if self.force_mode == "replay_raw_wrench" and force.shape[1] != 6:
                    raise ValueError(
                        "replay_raw_wrench requires all six force/torque columns, "
                        f"got {tuple(force.shape)}"
                    )
                seg_force = force[s:en, : self.wrench_dim].to(self.device)
                if self.wrench_dim == 6 and seg_force.shape[1] == 3:
                    seg_force = torch.cat(
                        (seg_force, torch.zeros_like(seg_force)),
                        dim=-1,
                    )
                bias_samples = max(0, int(getattr(tcfg, "dataset_force_bias_samples", 0)))
                if bias_samples > 0:
                    bias_count = min(bias_samples, raw_length)
                    seg_force = seg_force - seg_force[:bias_count].mean(dim=0, keepdim=True)
                if seg_force.shape[1] == 6:
                    torque_norm_samples.append(seg_force[:, 3:].norm(dim=-1))
                if chunk_length > 0:
                    wr = seg_force
                    if raw_length < storage_length:
                        wr = torch.cat(
                            (wr, wr[-1:].expand(storage_length - raw_length, -1)),
                            dim=0,
                        )
                else:
                    wr = self._resample_vec_sequence(seg_force, reference_length)
                if chunk_length == 0 and reference_length < length:
                    wr = torch.cat(
                        (wr, wr[-1:].expand(length - reference_length, -1)),
                        dim=0,
                    )
                wr_list.append(wr)
            if tcfg.dataset_max_trajs and len(pos_list) >= int(tcfg.dataset_max_trajs):
                break

        if not pos_list:
            raise RuntimeError(f"No usable episodes loaded from {path}")

        self._ds_pos = torch.stack(pos_list)  # (T, L, 3)
        self._ds_quat = torch.stack(quat_list)  # (T, L, 4)
        self._ds_phase = torch.stack(phase_list)  # (T, L), constant through padded tails
        self._ds_wrench = torch.stack(wr_list) if wr_list else None  # (T, L, wrench_dim) or None
        torque_clip_percentile = float(
            getattr(tcfg, "disturbance_torque_clip_percentile", 100.0)
        )
        if torque_norm_samples and torque_clip_percentile < 100.0:
            self._disturbance_torque_clip_norm = torch.quantile(
                torch.cat(torque_norm_samples),
                torque_clip_percentile / 100.0,
            ).item()
        self._ds_joint_pos = torch.stack(joint_list) if joint_list else None  # (T, L, 7)
        self._ds_lengths = (
            torch.tensor(raw_lengths, dtype=torch.long, device=self.device)
            if raw_lengths
            else torch.full((len(pos_list),), self._ds_pos.shape[1], dtype=torch.long, device=self.device)
        )
        self._ds_num = self._ds_pos.shape[0]

        pos_flat = self._ds_pos.reshape(-1, 3)
        box_min = [round(v, 3) for v in pos_flat.min(dim=0).values.tolist()]
        box_max = [round(v, 3) for v in pos_flat.max(dim=0).values.tolist()]
        force_str = ""
        if self._ds_wrench is not None:
            force_max = self._ds_wrench[..., :3].norm(dim=-1).max().item()
            force_str = f" | force |.| max {force_max:.2f} N"
            if self.wrench_dim == 6:
                torque_max = self._ds_wrench[..., 3:].norm(dim=-1).max().item()
                force_str += f" | torque |.| max {torque_max:.2f} Nm"
                if math.isfinite(self._disturbance_torque_clip_norm):
                    force_str += (
                        f" | applied torque p{torque_clip_percentile:g} clip "
                        f"{self._disturbance_torque_clip_norm:.3f} Nm"
                    )
        warp_str = ""
        if warp_prob > 0.0:
            warp_str = (
                f" | warp={warp_strategy} p={warp_prob:.2f}, "
                f"pos<={warp_pos_max * 1.0e3:.1f} mm, rot<={math.degrees(warp_rot_max):.1f} deg"
            )
            if warp_strategy == "constant_scale_shape":
                warp_str += f", shape={shape_fraction:.2f}, cycles={list(cycles_range)}"
        print(
            f"[FrankaRobustTrack] dataset mode: loaded {self._ds_num} trajectories from {path} "
            f"(reference L={reference_length}, runtime L={length}, "
            f"chunk_mode={'random_offset+tail_pad' if chunk_length > 0 else 'legacy_resample'}) | "
            f"pos box min {box_min} max {box_max}{force_str}{warp_str}"
        )

    def _sample_dataset_candidate_params(self, env_ids: torch.Tensor, num_candidates: int) -> dict:
        """Draw dataset trajectories and optional smooth spatial warps."""
        m = len(env_ids) * num_candidates
        tcfg = self.cfg.tracking
        warp_prob = float(getattr(tcfg, "dataset_warp_prob", 0.0))
        pos_max = float(getattr(tcfg, "dataset_warp_pos_max", 0.0))
        rot_max = float(getattr(tcfg, "dataset_warp_rot_max", 0.0))
        cycles_lo, cycles_hi = getattr(tcfg, "dataset_warp_cycles_range", [0.5, 1.5])
        warp_active = (torch.rand((m, 1), device=self.device) < warp_prob).float()

        # Sample the cone axes plus one perpendicular basis vector. The second
        # perpendicular vector is recovered with a cross product when evaluated.
        def sample_axis_and_perp():
            axis = torch.randn((m, 3), device=self.device)
            axis /= axis.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            perp = torch.randn((m, 3), device=self.device)
            perp -= (perp * axis).sum(dim=-1, keepdim=True) * axis
            perp /= perp.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            return axis, perp

        pos_dir, pos_u = sample_axis_and_perp()
        rot_dir, rot_u = sample_axis_and_perp()
        warp_pos = pos_dir * (torch.rand((m, 1), device=self.device) * pos_max) * warp_active
        warp_rot = rot_dir * (torch.rand((m, 1), device=self.device) * rot_max) * warp_active
        ds_idx = torch.randint(self._ds_num, (m,), device=self.device)
        chunk_length = int(getattr(tcfg, "dataset_chunk_length", 0))
        if chunk_length > 0:
            start_mode = str(
                getattr(tcfg, "dataset_chunk_start_mode", "episode_start")
            )
            if start_mode == "random":
                start_sampling = str(
                    getattr(tcfg, "dataset_chunk_start_sampling", "uniform")
                )
                if start_sampling == "adaptive":
                    probabilities = self._adaptive_dataset_probabilities()
                    pre_failure_window_steps = int(
                        round(
                            float(tcfg.adaptive_phase_pre_failure_window_s)
                            / self.reference_dt
                        )
                    )
                    ds_idx, ds_start = sample_variable_reference_starts(
                        probabilities,
                        self._adaptive_dataset_bin_sequence_ids,
                        self._adaptive_dataset_bin_starts,
                        self._adaptive_dataset_bin_ends,
                        num_samples=m,
                        pre_failure_window_steps=pre_failure_window_steps,
                    )
                elif start_sampling == "uniform":
                    # Every raw timestep is a valid reset point. History lookup
                    # repeats sample zero when t < H, while reference lookup
                    # repeats the final sample when the chunk reaches the tail.
                    lengths = self._ds_lengths[ds_idx]
                    ds_start = torch.floor(
                        torch.rand(m, device=self.device) * lengths.float()
                    ).long()
                elif start_sampling == "valid_steps":
                    # Exact rejection sampler for
                    #
                    #   p(start | episode) proportional to
                    #       min(chunk_length, episode_length - start).
                    #
                    # Uniform proposals followed by valid_steps/chunk_length
                    # acceptance preserve late-task starts but suppress heavily
                    # padded windows. Since most proposals have full weight, the
                    # loop normally completes in one or two passes.
                    lengths = self._ds_lengths[ds_idx]
                    ds_start = torch.zeros(m, dtype=torch.long, device=self.device)
                    pending = torch.ones(m, dtype=torch.bool, device=self.device)
                    while torch.any(pending):
                        pending_idx = pending.nonzero(as_tuple=False).squeeze(-1)
                        pending_lengths = lengths[pending_idx]
                        proposal = torch.floor(
                            torch.rand(pending_idx.numel(), device=self.device)
                            * pending_lengths.float()
                        ).long()
                        valid_steps = torch.minimum(
                            torch.full_like(proposal, chunk_length),
                            pending_lengths - proposal,
                        )
                        accepted = (
                            torch.rand(pending_idx.numel(), device=self.device)
                            * float(chunk_length)
                            < valid_steps.float()
                        )
                        accepted_idx = pending_idx[accepted]
                        ds_start[accepted_idx] = proposal[accepted]
                        pending[accepted_idx] = False
                else:
                    raise ValueError(
                        "tracking.dataset_chunk_start_sampling must be 'uniform', "
                        f"'valid_steps', or 'adaptive', got {start_sampling!r}"
                    )
            elif start_mode == "episode_start":
                ds_start = torch.zeros(m, dtype=torch.long, device=self.device)
            else:
                raise ValueError(
                    "tracking.dataset_chunk_start_mode must be 'episode_start' "
                    f"or 'random', got {start_mode!r}"
                )
        else:
            ds_start = torch.zeros(m, dtype=torch.long, device=self.device)
        return {
            "ds_idx": ds_idx,
            "ds_start": ds_start,
            "ds_warp_pos": warp_pos,
            "ds_warp_rot": warp_rot,
            "ds_warp_pos_u": pos_u,
            "ds_warp_rot_u": rot_u,
            "ds_warp_phase": 2.0 * math.pi * torch.rand((m, 1), device=self.device),
            "ds_warp_cycles": cycles_lo + (cycles_hi - cycles_lo) * torch.rand((m, 1), device=self.device),
        }

    def _apply_dataset_warp(
        self, pos: torch.Tensor, quat: torch.Tensor, params: dict, traj_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured smooth dataset trajectory warp.

        ``smooth_bump`` applies the original endpoint-preserving quartic envelope.
        ``constant_scale_shape`` rotates a fixed-norm offset along a cone, changing
        path shape without fading or compounding the perturbation.
        """
        warp_strategy = str(getattr(self.cfg.tracking, "dataset_warp_strategy", "smooth_bump"))
        if warp_strategy == "smooth_bump":
            phase = traj_phase.clamp(0.0, 1.0).unsqueeze(-1)
            blend = 16.0 * phase.square() * (1.0 - phase).square()
            pos_offset = blend * params["ds_warp_pos"]
            rot_vec = blend * params["ds_warp_rot"]
        else:
            pos_offset, rot_vec = self._constant_scale_shape_offsets(params, traj_phase)

        pos = pos + pos_offset
        angle = rot_vec.norm(dim=-1)
        axis = rot_vec / angle.unsqueeze(-1).clamp(min=1.0e-6)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        quat = torch_utils.quat_mul(delta_quat, quat)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return pos, quat

    def _constant_scale_shape_offsets(
        self, params: dict, traj_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-norm translation/rotation offsets with varying direction."""
        shape_fraction = float(getattr(self.cfg.tracking, "dataset_warp_shape_fraction", 0.0))
        axial_fraction = math.sqrt(max(0.0, 1.0 - shape_fraction**2))
        theta = (
            2.0 * math.pi * params["ds_warp_cycles"] * traj_phase.unsqueeze(-1)
            + params["ds_warp_phase"]
        )

        def cone_offset(base_vec: torch.Tensor, basis_u: torch.Tensor) -> torch.Tensor:
            magnitude = base_vec.norm(dim=-1, keepdim=True)
            axis = base_vec / magnitude.clamp(min=1.0e-6)
            basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
            direction = axial_fraction * axis + shape_fraction * (
                torch.cos(theta) * basis_u + torch.sin(theta) * basis_v
            )
            return magnitude * direction

        pos_offset = cone_offset(params["ds_warp_pos"], params["ds_warp_pos_u"])
        rot_vec = cone_offset(params["ds_warp_rot"], params["ds_warp_rot_u"])
        return pos_offset, rot_vec

    def _eval_dataset_pose(self, params: dict, elapsed_time: torch.Tensor):
        """Interpolate the selected demonstration trajectories at `elapsed_time`.

        `params["ds_idx"]` is (m,) trajectory indices; `elapsed_time` is (m, 1).
        The stored sequences already live on the native reference grid, so time maps to a
        fractional step index and is linearly (pos) / slerp (quat) interpolated.
        """
        idx = params["ds_idx"]
        chunk_length = int(getattr(self.cfg.tracking, "dataset_chunk_length", 0))
        if chunk_length > 0:
            start = params["ds_start"]
            effective_last = torch.minimum(
                torch.full_like(start, chunk_length - 1),
                self._ds_lengths[idx] - start - 1,
            ).clamp(min=0)
            rel_fidx = elapsed_time.squeeze(-1).div(self.reference_dt).clamp(min=0.0)
            rel_fidx = torch.minimum(rel_fidx, effective_last.float())
            fidx = start.float() + rel_fidx
            i0 = fidx.floor().long()
            i1 = torch.minimum(i0 + 1, start + effective_last)
            traj_phase = rel_fidx / effective_last.float().clamp(min=1.0)
        else:
            length = self._ds_pos.shape[1]
            fidx = (elapsed_time.squeeze(-1) / self.reference_dt).clamp(0.0, length - 1)
            i0 = fidx.floor().long()
            i1 = (i0 + 1).clamp(max=length - 1)
            phase_frac = fidx - i0.float()
            traj_phase = (
                self._ds_phase[idx, i0] * (1.0 - phase_frac)
                + self._ds_phase[idx, i1] * phase_frac
            )
        frac = (fidx - i0.float()).unsqueeze(-1)
        pos = self._ds_pos[idx, i0] * (1.0 - frac) + self._ds_pos[idx, i1] * frac
        quat = self._quat_slerp(self._ds_quat[idx, i0], self._ds_quat[idx, i1], frac.squeeze(-1))
        return self._apply_dataset_warp(pos, quat, params, traj_phase)

    def _sample_dataset_force(self, env_ids: torch.Tensor):
        """Fill the per-step target wrench for `env_ids` from their dataset trajectory."""
        ds_idx = self.traj_ds_idx[env_ids]
        chunk_length = int(getattr(self.cfg.tracking, "dataset_chunk_length", 0))
        if chunk_length > 0:
            source_idx, _ = self._dataset_chunk_indices(
                ds_idx, self.traj_ds_start[env_ids], self._traj_len
            )
            wr = self._ds_wrench[ds_idx.unsqueeze(1), source_idx]
        else:
            wr = self._ds_wrench[ds_idx]
        self.traj_wrench_buf[env_ids] = wr
        self.traj_applied_wrench_buf[env_ids] = 0.0
        if self.force_mode == "replay_raw_wrench":
            alpha = float(self.cfg.tracking.dataset_wrench_ema_alpha)
            self.traj_applied_wrench_buf[env_ids] = invert_ema_sequence(wr, alpha)
        else:
            # replay_disturbance now applies all available force and torque
            # components. Virtual contact overwrites this at runtime with its
            # contact-generated wrench, but retaining the complete target here
            # keeps reset/debug traces consistent.
            self.traj_applied_wrench_buf[env_ids, :, : self.wrench_dim] = wr
        if self.force_mode in ("replay_disturbance", "replay_raw_wrench"):
            if self.wrench_dim == 6 and math.isfinite(
                self._disturbance_torque_clip_norm
            ):
                torque = self.traj_applied_wrench_buf[env_ids, :, 3:]
                torque_norm = torque.norm(dim=-1, keepdim=True)
                torque *= (
                    self._disturbance_torque_clip_norm
                    / torque_norm.clamp_min(1.0e-12)
                ).clamp(max=1.0)
            self.traj_applied_wrench_buf[env_ids, :, :3] *= float(
                self.cfg.tracking.disturbance_force_scale
            )
            self.traj_applied_wrench_buf[env_ids, :, 3:] *= float(
                self.cfg.tracking.disturbance_torque_scale
            )
        threshold = float(self.cfg.tracking.dataset_contact_force_threshold)
        self.traj_contact_buf[env_ids] = wr[..., :3].norm(dim=-1) > threshold

    def _dataset_chunk_indices(
        self, ds_idx: torch.Tensor, ds_start: torch.Tensor, output_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source indices and local [0,1] phase for sampled chunks."""
        chunk_length = int(self.cfg.tracking.dataset_chunk_length)
        effective_last = torch.minimum(
            torch.full_like(ds_start, chunk_length - 1),
            self._ds_lengths[ds_idx] - ds_start - 1,
        ).clamp(min=0)
        rel = torch.arange(output_length, device=self.device).unsqueeze(0)
        rel = torch.minimum(rel, effective_last.unsqueeze(1))
        source_idx = ds_start.unsqueeze(1) + rel
        phase = rel.float() / effective_last.unsqueeze(1).float().clamp(min=1.0)
        return source_idx, phase

    def _seed_dataset_proprio_history(self, env_ids: torch.Tensor):
        """Prefill reset history on the policy-rate grid before each chunk.

        `_get_observations` will roll this buffer once and append the actual reset
        state at chunk time `t`. Fractional native-sample offsets preserve the
        original physical history span while adding high-rate intermediate frames.
        Dataset actions used Forge's different representation, so action-feedback
        slots remain zero rather than being mislabeled.
        """
        ds_idx = self.traj_ds_idx[env_ids]
        ds_start = self.traj_ds_start[env_ids]
        offsets = history_seed_offsets(
            self.obs_history_length, self.policy_steps_per_reference
        ).to(self.device)
        source_fidx = ds_start.float().unsqueeze(1) + offsets.unsqueeze(0)
        source_fidx = source_fidx.clamp(min=0.0)
        source_fidx = torch.minimum(
            source_fidx, (self._ds_lengths[ds_idx] - 1).float().unsqueeze(1)
        )
        batch_idx = ds_idx.unsqueeze(1)

        def interpolate(buffer: torch.Tensor, fidx: torch.Tensor) -> torch.Tensor:
            i0 = fidx.floor().long()
            i1 = (i0 + 1).clamp(max=buffer.shape[1] - 1)
            fraction = (fidx - i0.float()).unsqueeze(-1)
            value0 = buffer[batch_idx, i0]
            value1 = buffer[batch_idx, i1]
            return value0 * (1.0 - fraction) + value1 * fraction

        pos = interpolate(self._ds_pos, source_fidx)
        quat_i0 = source_fidx.floor().long()
        quat_i1 = (quat_i0 + 1).clamp(max=self._ds_quat.shape[1] - 1)
        quat_fraction = source_fidx - quat_i0.float()
        quat_shape = (len(env_ids), self.obs_history_length, 4)
        quat = self._quat_slerp(
            self._ds_quat[batch_idx, quat_i0].reshape(-1, 4),
            self._ds_quat[batch_idx, quat_i1].reshape(-1, 4),
            quat_fraction.reshape(-1),
        ).view(quat_shape)
        action = torch.zeros(
            (len(env_ids), self.obs_history_length, self.action_dim), device=self.device
        )
        history_parts = [pos, quat]
        if self.include_joint_angles:
            history_parts.append(interpolate(self._ds_joint_pos, source_fidx))
        if self.include_joint_velocities:
            previous_fidx = (
                source_fidx - 1.0 / float(self.policy_steps_per_reference)
            ).clamp(min=0.0)
            joint_pos = interpolate(self._ds_joint_pos, source_fidx)
            previous_joint_pos = interpolate(self._ds_joint_pos, previous_fidx)
            history_parts.append(
                (joint_pos - previous_joint_pos) / self.step_dt
            )
        if self.virtual_contact_enabled:
            history_parts.append(
                torch.zeros(
                    (len(env_ids), self.obs_history_length, self.force_feedback_dim),
                    device=self.device,
                )
            )
        history_parts.append(action)
        self.proprio_history[env_ids] = torch.cat(history_parts, dim=-1)

    def _traj_param_buffers(self) -> dict:
        """Active param-name -> per-env buffer-name map for the current tracking mode."""
        if self.cfg.tracking.mode == "dataset":
            return self._DS_TRAJ_PARAM_BUFFERS
        return self._TRAJ_PARAM_BUFFERS

    def _uses_recorded_dataset_reset(self) -> bool:
        """Whether this run uses random chunks with recorded state/history reset."""
        return (
            self.cfg.tracking.mode == "dataset"
            and int(getattr(self.cfg.tracking, "dataset_chunk_length", 0)) > 0
            and str(
                getattr(
                    self.cfg.tracking,
                    "dataset_chunk_start_mode",
                    "episode_start",
                )
            )
            == "random"
            and self._ds_joint_pos is not None
        )

    def _sample_candidate_params(self, env_ids: torch.Tensor, num_candidates: int) -> dict:
        """Sample `num_candidates` start-pose + trajectory param sets per env.

        Returns a dict of tensors, each with leading dim ``len(env_ids) *
        num_candidates`` flattened env-major (candidate c = b * k + j belongs to
        env ``env_ids[b]``). Env-specific centers (`nominal_start_pos/quat`) are
        broadcast across the `num_candidates` candidates of each env. Trajectory
        params for the inactive mode are filled with harmless defaults so
        `_assign_params_to_envs` can copy every key uniformly.
        """
        if self.cfg.tracking.mode == "dataset":
            return self._sample_dataset_candidate_params(env_ids, num_candidates)
        cfg = self.cfg.init
        tcfg = self.cfg.tracking
        device = self.device
        m = len(env_ids) * num_candidates
        env_rep = env_ids.repeat_interleave(num_candidates)

        if cfg.start_pos_box is not None:
            # Absolute box in the base/env-local frame: uniform in [min, max] per axis.
            box = torch.tensor(cfg.start_pos_box, device=device)  # (3, 2)
            lo, hi = box[:, 0], box[:, 1]
            start_pos = lo + (hi - lo) * torch.rand((m, 3), device=device)
        else:
            pos_noise = torch.tensor(cfg.start_pos_noise, device=device)
            offset = (2.0 * torch.rand((m, 3), device=device) - 1.0) * pos_noise
            start_pos = self.nominal_start_pos[env_rep] + offset

        axis = torch.randn((m, 3), device=device)
        axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
        angle = cfg.start_rot_noise * torch.rand(m, device=device)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        start_quat = torch_utils.quat_mul(delta_quat, self.nominal_start_quat[env_rep])

        params = {
            "start_pos": start_pos,
            "start_quat": start_quat,
            "dir": torch.zeros((m, 3), device=device),
            "speed": torch.zeros((m, 1), device=device),
            "length": torch.zeros((m, 1), device=device),
            "u": torch.zeros((m, 3), device=device),
            "v": torch.zeros((m, 3), device=device),
            "radius": torch.zeros((m, 1), device=device),
            "omega": torch.zeros((m, 1), device=device),
            "start_time": torch.zeros((m, 1), device=device),
        }
        params["u"][:, 0] = 1.0
        params["v"][:, 1] = 1.0

        if tcfg.mode in ("line", "line_fixed"):
            if tcfg.mode == "line_fixed":
                # Fixed direction: same unit vector for every env / candidate.
                direction = torch.tensor(tcfg.line_dir, device=device, dtype=torch.float32)
                direction = direction / torch.clamp(torch.linalg.norm(direction), min=1.0e-6)
                direction = direction.unsqueeze(0).expand(m, 3).contiguous()
            else:
                direction = torch.randn((m, 3), device=device)
                direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1.0e-6)
            params["dir"] = direction
            length_lo, length_hi = tcfg.line_length_range
            length = length_lo + (length_hi - length_lo) * torch.rand((m, 1), device=device)
            params["length"] = length
            # Speed is not sampled independently: the line is traversed exactly
            # once over the full episode, so speed = length / episode_length. This
            # keeps the reference moving for the whole episode (no early saturation)
            # regardless of the sampled length.
            params["speed"] = length / self.max_episode_length_s
        elif tcfg.mode == "circle":
            # Random orthonormal plane basis (u, v) via Gram-Schmidt.
            u = torch.randn((m, 3), device=device)
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            v = torch.randn((m, 3), device=device)
            v = v - (v * u).sum(dim=-1, keepdim=True) * u
            v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1.0e-6)
            params["u"] = u
            params["v"] = v
            radius_lo, radius_hi = tcfg.circle_radius_range
            params["radius"] = radius_lo + (radius_hi - radius_lo) * torch.rand((m, 1), device=device)
            speed_lo, speed_hi = tcfg.circle_speed_range
            omega = speed_lo + (speed_hi - speed_lo) * torch.rand((m, 1), device=device)
            direction_sign = torch.where(
                torch.rand((m, 1), device=device) < 0.5,
                -torch.ones((m, 1), device=device),
                torch.ones((m, 1), device=device),
            )
            params["omega"] = omega * direction_sign
            start_time_lo, start_time_hi = tcfg.circle_start_time_range
            if start_time_lo == start_time_hi:
                params["start_time"] = torch.full(
                    (m, 1), float(start_time_lo), device=device
                )
            elif tcfg.circle_start_time_sampling == "valid_steps":
                # Sample with density proportional to the amount of the episode
                # that precedes terminal reference padding:
                #
                #   weight(t) = min(episode_duration, reference_duration - t)
                #
                # This is the continuous-time equivalent of weighting candidate
                # starts by their number of valid (non-pad) policy steps. The
                # inverse CDF has a uniform-density prefix and a linearly
                # decreasing tail, so it is exact and requires no rejection loop.
                params["start_time"] = valid_step_weighted_start_times(
                    torch.rand((m, 1), device=device),
                    start_time_lo=start_time_lo,
                    start_time_hi=start_time_hi,
                    reference_duration=tcfg.circle_reference_duration_s,
                    episode_duration=self.max_episode_length_s,
                )
            elif tcfg.circle_start_time_sampling == "adaptive":
                probabilities = self._adaptive_phase_probabilities(
                    start_time_lo=start_time_lo,
                    start_time_hi=start_time_hi,
                )
                params["start_time"] = sample_adaptive_phase_start_times(
                    probabilities,
                    num_samples=m,
                    bin_size_s=tcfg.adaptive_phase_bin_size_s,
                    start_time_lo=start_time_lo,
                    start_time_hi=start_time_hi,
                    reference_duration=tcfg.circle_reference_duration_s,
                    pre_failure_window_s=(
                        tcfg.adaptive_phase_pre_failure_window_s
                    ),
                )
            else:
                params["start_time"] = start_time_lo + (
                    start_time_hi - start_time_lo
                ) * torch.rand((m, 1), device=device)
        else:
            raise ValueError(f"Unsupported tracking mode: {tcfg.mode}")

        rot_axis = torch.randn((m, 3), device=device)
        rot_axis = rot_axis / torch.clamp(torch.linalg.norm(rot_axis, dim=-1, keepdim=True), min=1.0e-6)
        params["rot_axis"] = rot_axis
        rot_speed_lo, rot_speed_hi = tcfg.rot_speed_range
        params["rot_speed"] = rot_speed_lo + (rot_speed_hi - rot_speed_lo) * torch.rand((m, 1), device=device)
        rot_angle_lo, rot_angle_hi = tcfg.rot_angle_range
        params["rot_angle"] = rot_angle_lo + (rot_angle_hi - rot_angle_lo) * torch.rand((m, 1), device=device)
        return params

    _TRAJ_PARAM_BUFFERS = {
        "start_pos": "traj_start_pos",
        "start_quat": "traj_start_quat",
        "dir": "traj_dir",
        "speed": "traj_speed",
        "length": "traj_length",
        "u": "traj_u",
        "v": "traj_v",
        "radius": "traj_radius",
        "omega": "traj_omega",
        "start_time": "traj_start_time",
        "rot_axis": "traj_rot_axis",
        "rot_speed": "traj_rot_speed",
        "rot_angle": "traj_rot_angle",
    }

    # Dataset poses are gathered from the loaded tensors on demand, then bent by
    # the per-env warp vectors (zero vectors mean no augmentation).
    _DS_TRAJ_PARAM_BUFFERS = {
        "ds_idx": "traj_ds_idx",
        "ds_start": "traj_ds_start",
        "ds_warp_pos": "traj_ds_warp_pos",
        "ds_warp_rot": "traj_ds_warp_rot",
        "ds_warp_pos_u": "traj_ds_warp_pos_u",
        "ds_warp_rot_u": "traj_ds_warp_rot_u",
        "ds_warp_phase": "traj_ds_warp_phase",
        "ds_warp_cycles": "traj_ds_warp_cycles",
    }

    def _assign_params_to_envs(self, env_ids: torch.Tensor, params: dict, cand_idx: torch.Tensor):
        """Copy the selected candidate params (`cand_idx`) into the per-env traj buffers."""
        for key, buf_name in self._traj_param_buffers().items():
            getattr(self, buf_name)[env_ids] = params[key][cand_idx]

    def _eval_pose_from_params(self, params: dict, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` for the given trajectory params.

        `params` maps the trajectory keys to tensors with a shared leading dim `M`;
        `elapsed_time` is (M, 1). Unless an analytic circle has a nonzero
        `start_time`, the trajectory passes through `start_pos`/`start_quat` at
        ``elapsed_time == 0``.
        """
        mode = self.cfg.tracking.mode
        if mode == "dataset":
            return self._eval_dataset_pose(params, elapsed_time)
        start_pos = params["start_pos"]
        start_quat = params["start_quat"]
        if mode in ("line", "line_fixed"):
            displacement = torch.minimum(params["speed"] * elapsed_time, params["length"])
            pos = start_pos + params["dir"] * displacement
        elif mode == "circle":
            circle_time = torch.clamp(
                elapsed_time + params["start_time"],
                max=float(
                    getattr(
                        self.cfg.tracking,
                        "circle_reference_duration_s",
                        self.max_episode_length_s,
                    )
                ),
            )
            theta = params["omega"] * circle_time
            pos = start_pos + params["radius"] * (
                (torch.cos(theta) - 1.0) * params["u"] + torch.sin(theta) * params["v"]
            )
        else:
            raise ValueError(f"Unsupported tracking mode: {mode}")

        orientation_time = circle_time if mode == "circle" else elapsed_time
        sweep_angle = torch.minimum(
            params["rot_speed"] * orientation_time, params["rot_angle"]
        ).squeeze(-1)
        delta_quat = torch_utils.quat_from_angle_axis(sweep_angle, params["rot_axis"])
        quat = torch_utils.quat_mul(delta_quat, start_quat)
        return pos, quat

    def _command_pose_at(self, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` (shape (num_envs, 1))."""
        params = {key: getattr(self, buf_name) for key, buf_name in self._traj_param_buffers().items()}
        return self._eval_pose_from_params(params, elapsed_time)

    def _build_trajectory_buffer(self, env_ids: torch.Tensor):
        """Sample the analytic trajectory onto the native grid for `env_ids`.

        Grid times are `arange(self._traj_len) * reference_dt`. Async runtime
        interpolates the active command while lookahead reads native knots.
        """
        if self.cfg.tracking.mode == "dataset":
            # Sequences are already prepared on the native grid. Apply the
            # accepted candidate's warp in one batch so runtime lookup stays cheap.
            params = {
                key: getattr(self, buf_name)[env_ids] for key, buf_name in self._DS_TRAJ_PARAM_BUFFERS.items()
            }
            chunk_length = int(getattr(self.cfg.tracking, "dataset_chunk_length", 0))
            if chunk_length > 0:
                source_idx, traj_phase = self._dataset_chunk_indices(
                    params["ds_idx"], params["ds_start"], self._traj_len
                )
                batch_idx = params["ds_idx"].unsqueeze(1)
                pos = self._ds_pos[batch_idx, source_idx]
                quat = self._ds_quat[batch_idx, source_idx]
            else:
                pos = self._ds_pos[params["ds_idx"]]
                quat = self._ds_quat[params["ds_idx"]]
                traj_phase = self._ds_phase[params["ds_idx"]]
            flat_params = {
                "ds_warp_pos": params["ds_warp_pos"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_rot": params["ds_warp_rot"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_pos_u": params["ds_warp_pos_u"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_rot_u": params["ds_warp_rot_u"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_phase": params["ds_warp_phase"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 1),
                "ds_warp_cycles": params["ds_warp_cycles"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 1),
            }
            pos, quat = self._apply_dataset_warp(
                pos.reshape(-1, 3), quat.reshape(-1, 4), flat_params, traj_phase.reshape(-1)
            )
            self.traj_pos_buf[env_ids] = pos.view(len(env_ids), self._traj_len, 3)
            self.traj_quat_buf[env_ids] = quat.view(len(env_ids), self._traj_len, 4)
            return
        for i in range(self._traj_len):
            elapsed_time = torch.full((self.num_envs, 1), i * self.reference_dt, device=self.device)
            pos_i, quat_i = self._command_pose_at(elapsed_time)
            self.traj_pos_buf[env_ids, i] = pos_i[env_ids]
            self.traj_quat_buf[env_ids, i] = quat_i[env_ids]

    def _traj_pose_at_index(self, idx: torch.Tensor):
        """Gather the discretized reference pose at per-env step index `idx` (num_envs,)."""
        idx = idx.clamp(0, self._traj_len - 1)
        return self.traj_pos_buf[self._env_arange, idx], self.traj_quat_buf[self._env_arange, idx]

    def _traj_wrench_at_index(self, idx: torch.Tensor):
        """Gather the target wrench in its configured dataset convention."""
        idx = idx.clamp(0, self._traj_len - 1)
        return self.traj_wrench_buf[self._env_arange, idx]

    def _policy_step_to_reference_index(
        self, policy_step: torch.Tensor
    ) -> torch.Tensor:
        """Map policy steps to native reference samples using zero-order hold."""
        return policy_step_to_reference_index(
            policy_step, self.policy_steps_per_reference
        )

    def _reference_coordinates(
        self,
        policy_step: torch.Tensor,
        completed_physics_substeps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return native reference endpoints and interpolation phase.

        ``policy_step`` is the environment counter at the start of a policy
        action. ``completed_physics_substeps`` selects a point inside that
        action. At a policy boundary it is zero; inside ``_apply_action`` it is
        1..decimation so virtual contact follows the native segment at 120 Hz.
        """
        i0, i1, phase = reference_coordinates(
            policy_step,
            policy_decimation=int(self.cfg.decimation),
            reference_decimation=self.reference_decimation,
            completed_physics_substeps=completed_physics_substeps,
        )
        i0 = i0.clamp(0, self._traj_len - 1)
        i1 = i1.clamp(0, self._traj_len - 1)
        return i0, i1, phase

    def _traj_pose_at_policy_step(
        self,
        policy_step: torch.Tensor,
        completed_physics_substeps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linearly interpolate position and slerp rotation on the native grid."""
        i0, i1, phase = self._reference_coordinates(
            policy_step, completed_physics_substeps
        )
        pos0, quat0 = self._traj_pose_at_index(i0)
        pos1, quat1 = self._traj_pose_at_index(i1)
        blend = phase.unsqueeze(-1)
        pos = pos0 * (1.0 - blend) + pos1 * blend
        quat = self._quat_slerp(quat0, quat1, phase)
        return pos, quat

    def _traj_wrench_at_policy_step(
        self,
        policy_step: torch.Tensor,
        completed_physics_substeps: int = 0,
    ) -> torch.Tensor:
        """Linearly interpolate the virtual-contact wrench goal."""
        i0, i1, phase = self._reference_coordinates(
            policy_step, completed_physics_substeps
        )
        wrench0 = self._traj_wrench_at_index(i0)
        wrench1 = self._traj_wrench_at_index(i1)
        blend = phase.unsqueeze(-1)
        return wrench0 * (1.0 - blend) + wrench1 * blend

    def _smooth_osc(self, tau: torch.Tensor, num_components: int = 3) -> torch.Tensor:
        """A smooth per-step signal in [-1, 1] built from a few random sinusoids.

        `tau` is (n, L) time-since-onset in seconds. Frequencies are drawn in
        `tracking.force_osc_freq_range`, phases uniformly, and the components are
        combined with weights that sum to 1 so the weighted sum of unit sines stays
        within [-1, 1]. The result varies continuously in time (no jumps).
        """
        n = tau.shape[0]
        device = tau.device
        freq_lo, freq_hi = self.cfg.tracking.force_osc_freq_range
        freq = freq_lo + (freq_hi - freq_lo) * torch.rand((n, num_components), device=device)
        phase = 2.0 * math.pi * torch.rand((n, num_components), device=device)
        weights = torch.rand((n, num_components), device=device)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        ang = 2.0 * math.pi * freq.unsqueeze(-1) * tau.unsqueeze(1) + phase.unsqueeze(-1)  # (n, K, L)
        return (weights.unsqueeze(-1) * torch.sin(ang)).sum(dim=1)  # (n, L)

    def _sample_force_trajectory(self, env_ids: torch.Tensor):
        """Sample a per-step target-wrench sequence for `env_ids` into `traj_wrench_buf`.

        Models a handful of contact events: the force is zero for most of the
        episode, punctuated by `force_num_bands_range` bands. The episode is split
        into that many equal segments (per env), and each segment holds one band
        placed at a random onset with a sampled duration, so bands never overlap and
        stay spread out. Each band switches on suddenly (step, no ramp) and, while
        active, varies smoothly: its magnitude oscillates continuously in
        ``[0.5, 1.0] * peak`` (peak sampled per band in `force_mag_range`) and its
        direction drifts smoothly around a random base direction by
        ``force_dir_drift``. With probability ``1 - force_prob`` the env sees no
        contact at all. Independent of the pose-reachability sampling since a wrist
        force does not change IK reachability.
        """
        tcfg = self.cfg.tracking
        if tcfg.mode == "dataset" and tcfg.dataset_force_key and self._ds_wrench is not None:
            # Dataset mode: replay the demonstration's own per-step contact force.
            self._sample_dataset_force(env_ids)
            return
        device = self.device
        n = len(env_ids)

        band_lo, band_hi = tcfg.force_num_bands_range
        max_bands = int(band_hi)
        num_bands = torch.randint(int(band_lo), max_bands + 1, (n, 1), device=device)  # (n, 1)
        seg_len = float(self.max_episode_length) / num_bands.float()  # (n, 1) steps per segment

        steps = torch.arange(self._traj_len, device=device).unsqueeze(0)  # (1, L)
        wrench = torch.zeros((n, self._traj_len, self.wrench_dim), device=device)
        contact = torch.zeros((n, self._traj_len), dtype=torch.bool, device=device)

        dur_lo, dur_hi = tcfg.force_duration_range
        for b in range(max_bands):
            use_b = b < num_bands  # (n, 1) bool: this env actually has a b-th band

            # Duration (steps), capped so the band fits within its own segment.
            duration_s = dur_lo + (dur_hi - dur_lo) * torch.rand((n, 1), device=device)
            duration_steps = (duration_s / self.reference_dt).round().clamp(min=1.0)
            duration_steps = torch.minimum(duration_steps, seg_len.clamp(min=1.0))  # (n, 1)

            # Random onset inside segment b: [b * seg_len, (b + 1) * seg_len - duration].
            seg_start = b * seg_len
            onset = seg_start + torch.rand((n, 1), device=device) * (seg_len - duration_steps).clamp(min=0.0)

            active = (steps >= onset) & (steps < onset + duration_steps) & use_b  # (n, L)

            # Smooth time-varying profile within this band (time measured from onset).
            tau = (steps - onset).float() * self.reference_dt  # (n, L)
            mag_lo, mag_hi = tcfg.force_mag_range
            peak = (mag_lo + (mag_hi - mag_lo) * torch.rand((n, 1), device=device)).clamp(max=20.0)
            mag = peak * (0.75 + 0.25 * self._smooth_osc(tau))  # (n, L)

            base_dir = torch.randn((n, 3), device=device)
            base_dir = base_dir / torch.clamp(torch.linalg.norm(base_dir, dim=-1, keepdim=True), min=1.0e-6)
            perp1, perp2 = self._orthonormal_basis(base_dir)
            s1 = self._smooth_osc(tau).unsqueeze(-1)
            s2 = self._smooth_osc(tau).unsqueeze(-1)
            dir_raw = base_dir.unsqueeze(1) + tcfg.force_dir_drift * (
                perp1.unsqueeze(1) * s1 + perp2.unsqueeze(1) * s2
            )  # (n, L, 3)
            direction = dir_raw / torch.clamp(torch.linalg.norm(dir_raw, dim=-1, keepdim=True), min=1.0e-6)

            active_f = active.float()
            # Segments are disjoint, so bands never overlap: accumulate is safe.
            wrench[..., :3] = wrench[..., :3] + direction * (mag * active_f).unsqueeze(-1)
            contact = contact | active

        if tcfg.force_prob < 1.0:
            keep = torch.rand((n, 1), device=device) < tcfg.force_prob  # (n, 1)
            wrench = wrench * keep.unsqueeze(-1)
            contact = contact & keep

        self.traj_wrench_buf[env_ids] = wrench
        self.traj_applied_wrench_buf[env_ids] = 0.0
        self.traj_applied_wrench_buf[env_ids, :, : self.wrench_dim] = wrench
        self.traj_contact_buf[env_ids] = contact

    def _sample_virtual_contact_range(
        self,
        values,
        count: int,
        *,
        log_uniform: bool = False,
    ) -> torch.Tensor:
        """Sample one scalar contact parameter for each resetting environment."""
        low, high = (float(value) for value in values)
        unit = torch.rand((count, 1), device=self.device)
        if log_uniform:
            return torch.exp(math.log(low) + (math.log(high) - math.log(low)) * unit)
        return low + (high - low) * unit

    def _sample_virtual_contact_parameters(self, env_ids: torch.Tensor):
        """Sample or restore episode-constant virtual-contact parameters."""
        count = len(env_ids)
        if count == 0:
            return
        tcfg = self.cfg.tracking
        if bool(tcfg.virtual_contact_randomize_parameters):
            self.virtual_contact_power_exponent[env_ids] = self._sample_virtual_contact_range(
                tcfg.power_law_exponent_range, count
            )
            self.virtual_contact_reference_penetration[env_ids] = (
                self._sample_virtual_contact_range(
                    tcfg.power_law_reference_penetration_range,
                    count,
                    log_uniform=True,
                )
            )
            self.virtual_contact_dissipation[env_ids] = self._sample_virtual_contact_range(
                tcfg.hunt_crossley_dissipation_range, count
            )
            self.virtual_contact_normal_tilt[env_ids] = self._sample_virtual_contact_range(
                tcfg.virtual_contact_normal_tilt_range, count
            )
            self.virtual_contact_rotational_power_exponent[env_ids] = (
                self._sample_virtual_contact_range(
                    tcfg.virtual_contact_rotational_power_exponent_range, count
                )
            )
            self.virtual_contact_rotational_reference_penetration[env_ids] = (
                self._sample_virtual_contact_range(
                    tcfg.virtual_contact_rotational_reference_penetration_range,
                    count,
                    log_uniform=True,
                )
            )
            self.virtual_contact_rotational_damping[env_ids] = (
                self._sample_virtual_contact_range(
                    tcfg.virtual_contact_rotational_damping_range, count
                )
            )
        else:
            self.virtual_contact_power_exponent[env_ids] = float(tcfg.power_law_exponent)
            self.virtual_contact_reference_penetration[env_ids] = float(
                tcfg.power_law_reference_penetration
            )
            self.virtual_contact_dissipation[env_ids] = float(
                tcfg.hunt_crossley_dissipation
            )
            self.virtual_contact_normal_tilt[env_ids] = 0.0
            self.virtual_contact_rotational_power_exponent[env_ids] = float(
                tcfg.virtual_contact_rotational_power_exponent
            )
            self.virtual_contact_rotational_reference_penetration[env_ids] = float(
                tcfg.virtual_contact_rotational_reference_penetration
            )
            self.virtual_contact_rotational_damping[env_ids] = float(
                tcfg.virtual_contact_rotational_damping
            )
        if bool(
            tcfg.virtual_contact_randomize_parameters
            or tcfg.virtual_contact_randomize_surface_offset
        ):
            self.virtual_contact_surface_offset[env_ids] = self._sample_virtual_contact_range(
                tcfg.virtual_contact_surface_offset_range, count
            )
        else:
            self.virtual_contact_surface_offset[env_ids] = 0.0

    def _sample_virtual_contact_surface(self, env_ids: torch.Tensor):
        """Construct a randomized virtual plane at every reference timestep.

        A centered moving average stabilizes the demonstrated force direction.
        For the selected elastic coefficient k_p and exponent p, each plane is
        placed (|F_goal| / k_p) ** (1/p) above the reference EE pose along that
        direction. The nominal plane therefore reproduces the target magnitude;
        the optional episode-level offset and tilt are applied afterward so the
        policy must correct the resulting hidden geometry error.
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return
        tcfg = self.cfg.tracking
        self._sample_virtual_contact_parameters(env_ids)
        wrench = self.traj_wrench_buf[env_ids]
        if self.dataset_wrench_convention == "sensor_child_joint_reaction":
            sensor_quat_in_tool = self.force_sensor_quat_in_tool[env_ids].unsqueeze(1).expand(
                -1, self._traj_len, -1
            )
            sensor_quat_world = math_utils.quat_mul(
                self.traj_quat_buf[env_ids], sensor_quat_in_tool
            )
            external_wrench_world = sensor_reaction_to_world_external(
                wrench, sensor_quat_world
            )
        else:
            external_wrench_world = wrench
        reference_force_scale = float(tcfg.virtual_contact_reference_force_scale)
        reference_torque_scale = float(tcfg.virtual_contact_reference_torque_scale)
        scaled_external_wrench_world = external_wrench_world.clone()
        scaled_external_wrench_world[..., :3] *= reference_force_scale
        scaled_external_wrench_world[..., 3:] *= reference_torque_scale
        scaled_force_world = scaled_external_wrench_world[..., :3]
        if str(tcfg.contact_model) == "power_law":
            contact_exponent = self.virtual_contact_power_exponent[env_ids]
            contact_coefficient = power_law_coefficient_from_reference(
                reference_force=float(tcfg.power_law_reference_force),
                reference_penetration=self.virtual_contact_reference_penetration[
                    env_ids
                ],
                exponent=contact_exponent,
            )
        elif str(tcfg.contact_model) == "exponential":
            reference_force = float(tcfg.power_law_reference_force)
            reference_penetration = float(tcfg.power_law_reference_penetration)
            sharpness = float(tcfg.exponential_sharpness)
            contact_coefficient = reference_force / reference_penetration
            contact_exponent = 1.0
        else:
            contact_coefficient = float(tcfg.virtual_contact_plane_stiffness)
            contact_exponent = 1.0
        if str(tcfg.contact_model) == "exponential":
            surface_normal = smooth_force_directions(
                scaled_force_world,
                window=int(tcfg.virtual_contact_direction_smoothing_window),
                valid_force_threshold=float(tcfg.virtual_contact_goal_threshold),
            )
            scaled_magnitude = torch.linalg.norm(scaled_force_world, dim=-1, keepdim=True)
            penetration = (reference_penetration / sharpness) * torch.log1p(
                (scaled_magnitude / reference_force) * math.expm1(sharpness)
            )
            surface_point = self.traj_pos_buf[env_ids] + penetration * surface_normal
        else:
            surface_point, surface_normal, _ = construct_reference_plane_trajectory(
                reference_position=self.traj_pos_buf[env_ids],
                target_force=scaled_force_world,
                plane_stiffness=contact_coefficient,
                direction_smoothing_window=int(tcfg.virtual_contact_direction_smoothing_window),
                direction_force_threshold=float(tcfg.virtual_contact_goal_threshold),
                power_law_exponent=contact_exponent,
            )
        if bool(
            tcfg.virtual_contact_randomize_parameters
            or tcfg.virtual_contact_randomize_surface_offset
        ):
            base_normal = surface_normal[:, 0]
            random_axis = torch.randn_like(base_normal)
            random_axis = random_axis - (
                random_axis * base_normal
            ).sum(dim=-1, keepdim=True) * base_normal
            axis_norm = torch.linalg.norm(random_axis, dim=-1, keepdim=True)
            fallback_axis, _ = self._orthonormal_basis(base_normal)
            random_axis = torch.where(
                axis_norm > 1.0e-6,
                random_axis / axis_norm.clamp_min(1.0e-6),
                fallback_axis,
            )
            tilt_rotation_vector = (
                random_axis * self.virtual_contact_normal_tilt[env_ids]
            )
            surface_point, surface_normal = perturb_reference_plane_trajectory(
                surface_point,
                surface_normal,
                normal_offset=self.virtual_contact_surface_offset[env_ids],
                tilt_rotation_vector=tilt_rotation_vector,
            )
        # Plane placement uses the converted/scaled world external force, while
        # observations and metrics retain the original recorded sensor reaction.
        stiffness = torch.as_tensor(
            contact_coefficient, dtype=surface_point.dtype, device=self.device
        )
        if stiffness.ndim == 0:
            stiffness = stiffness.expand(n_envs, 1).clone()
        else:
            stiffness = stiffness.reshape(n_envs, 1)
        damping_ratio = float(tcfg.virtual_contact_damping_ratio)
        effective_mass = max(float(tcfg.virtual_contact_effective_mass), 1.0e-6)
        if str(tcfg.contact_model) == "linear":
            damping = 2.0 * damping_ratio * torch.sqrt(
                stiffness.clamp(min=0.0) * effective_mass
            )
        else:
            # Hunt-Crossley uses its own multiplicative dissipation coefficient;
            # the additive linear damping coefficient is intentionally inactive.
            damping = torch.zeros_like(stiffness)

        self.traj_surface_point_buf[env_ids] = surface_point
        self.traj_surface_normal_buf[env_ids] = surface_normal
        self.traj_surface_velocity_buf[env_ids] = finite_difference_surface_velocity(
            surface_point, timestep=self.reference_dt
        )
        target_torque_world = (
            scaled_external_wrench_world[..., 3:]
            if self.wrench_dim == 6
            else torch.zeros_like(scaled_force_world)
        )
        rotational_power_exponent = (
            self.virtual_contact_rotational_power_exponent[env_ids]
            if tcfg.virtual_contact_torque_model == "orientation_power_law"
            else 1.0
        )
        rotational_coefficient = (
            power_law_coefficient_from_reference(
                reference_force=float(
                    tcfg.virtual_contact_rotational_reference_torque
                ),
                reference_penetration=self.virtual_contact_rotational_reference_penetration[
                    env_ids
                ],
                exponent=rotational_power_exponent,
            )
            if tcfg.virtual_contact_torque_model == "orientation_power_law"
            else float(tcfg.virtual_contact_rotational_stiffness)
        )
        if isinstance(rotational_coefficient, torch.Tensor):
            self.virtual_contact_rotational_coefficient[env_ids] = rotational_coefficient
        else:
            self.virtual_contact_rotational_coefficient[env_ids] = float(
                rotational_coefficient
            )
        rotational_equilibrium_quat = construct_rotational_equilibrium_trajectory(
            reference_quat_world=self.traj_quat_buf[env_ids],
            target_torque_world=target_torque_world,
            rotational_stiffness=rotational_coefficient,
            power_law_exponent=rotational_power_exponent,
            torque_scale=float(tcfg.virtual_contact_torque_scale),
            torque_cap=float(tcfg.virtual_contact_torque_cap),
        )
        self.traj_rotational_equilibrium_quat_buf[env_ids] = rotational_equilibrium_quat
        self.traj_rotational_equilibrium_angvel_buf[env_ids] = (
            finite_difference_quaternion_angular_velocity(
                rotational_equilibrium_quat,
                timestep=self.reference_dt,
            )
        )
        self.traj_virtual_target_wrench_buf[env_ids] = scaled_external_wrench_world
        self.virtual_surface_stiffness[env_ids] = stiffness
        self.virtual_surface_damping[env_ids] = damping
        self.virtual_surface_distance[env_ids] = 1.0
        self.virtual_contact_force[env_ids] = 0.0
        self.virtual_contact_torque[env_ids] = 0.0
        self.virtual_target_force[env_ids] = 0.0
        self.virtual_target_torque[env_ids] = 0.0
        self.force_sensor_raw[env_ids] = 0.0
        self.force_sensor_smooth[env_ids] = 0.0
        self.force_sensor_obs[env_ids] = 0.0
        bias_values = [max(0.0, float(tcfg.force_sensor_bias_range))] * 3
        if self.wrench_dim == 6:
            bias_values += [max(0.0, float(tcfg.torque_sensor_bias_range))] * 3
        bias_scale = torch.tensor(bias_values, device=self.device)
        self.force_sensor_bias[env_ids] = bias_scale * (
            2.0 * torch.rand((n_envs, self.wrench_dim), device=self.device) - 1.0
        )
        self.current_contact[env_ids] = False

        goal_mag = torch.linalg.norm(wrench[..., :3], dim=-1)
        self.traj_contact_buf[env_ids] = goal_mag > float(tcfg.dataset_contact_force_threshold)

    def _set_virtual_contact_surface_at_index(self, idx: torch.Tensor):
        """Select and hold one native-reference virtual-contact state."""
        idx = idx.clamp(0, self._traj_len - 1)
        self.virtual_surface_point[:] = self.traj_surface_point_buf[self._env_arange, idx]
        self.virtual_surface_normal[:] = self.traj_surface_normal_buf[self._env_arange, idx]
        self.virtual_surface_velocity[:] = self.traj_surface_velocity_buf[self._env_arange, idx]
        self.virtual_rotational_equilibrium_quat[:] = (
            self.traj_rotational_equilibrium_quat_buf[self._env_arange, idx]
        )
        self.virtual_rotational_equilibrium_angvel[:] = (
            self.traj_rotational_equilibrium_angvel_buf[self._env_arange, idx]
        )
        target_wrench = self.traj_virtual_target_wrench_buf[self._env_arange, idx]
        self.virtual_target_force[:] = target_wrench[:, :3]
        if self.wrench_dim == 6:
            self.virtual_target_torque[:] = target_wrench[:, 3:]
        else:
            self.virtual_target_torque.zero_()
        self.target_wrench[:] = self._traj_wrench_at_index(idx)

    def _set_virtual_contact_surface_at_policy_step(
        self,
        policy_step: torch.Tensor,
        completed_physics_substeps: int = 0,
    ):
        """Select the virtual-contact state using interpolation or native ZOH."""
        i0, i1, phase = virtual_contact_reference_coordinates(
            policy_step,
            policy_decimation=int(self.cfg.decimation),
            reference_decimation=self.reference_decimation,
            completed_physics_substeps=completed_physics_substeps,
            interpolate=bool(
                self.cfg.tracking.virtual_contact_interpolate_reference
            ),
        )
        i0 = i0.clamp(0, self._traj_len - 1)
        i1 = i1.clamp(0, self._traj_len - 1)
        blend = phase.unsqueeze(-1)

        def interpolate(buffer):
            value0 = buffer[self._env_arange, i0]
            value1 = buffer[self._env_arange, i1]
            return value0 * (1.0 - blend) + value1 * blend

        self.virtual_surface_point[:] = interpolate(self.traj_surface_point_buf)
        normal = interpolate(self.traj_surface_normal_buf)
        self.virtual_surface_normal[:] = normal / normal.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        self.virtual_surface_velocity[:] = interpolate(
            self.traj_surface_velocity_buf
        )
        equilibrium_q0 = self.traj_rotational_equilibrium_quat_buf[
            self._env_arange, i0
        ]
        equilibrium_q1 = self.traj_rotational_equilibrium_quat_buf[
            self._env_arange, i1
        ]
        self.virtual_rotational_equilibrium_quat[:] = quaternion_nlerp(
            equilibrium_q0,
            equilibrium_q1,
            blend,
        )
        self.virtual_rotational_equilibrium_angvel[:] = interpolate(
            self.traj_rotational_equilibrium_angvel_buf
        )
        target_wrench = interpolate(self.traj_virtual_target_wrench_buf)
        self.virtual_target_force[:] = target_wrench[:, :3]
        if self.wrench_dim == 6:
            self.virtual_target_torque[:] = target_wrench[:, 3:]
        else:
            self.virtual_target_torque.zero_()
        self.target_wrench[:] = interpolate(self.traj_wrench_buf)

    def _update_virtual_contact_force(
        self, completed_physics_substeps: int = 0
    ):
        """Compute contact at the requested instant within the policy transition.

        The final physics substep uses the same instant as the subsequent
        post-action pose/wrench reward, avoiding a plane[t] versus wrench[t+1]
        comparison.
        """
        self._set_virtual_contact_surface_at_policy_step(
            self.episode_length_buf, completed_physics_substeps
        )
        distance, contact_force = compute_virtual_plane_contact(
            position=self.fingertip_midpoint_pos,
            linear_velocity=self.fingertip_midpoint_linvel,
            surface_point=self.virtual_surface_point,
            surface_normal=self.virtual_surface_normal,
            stiffness=self.virtual_surface_stiffness,
            damping=self.virtual_surface_damping,
            transition_width=float(self.cfg.tracking.virtual_contact_transition_width),
            force_cap=float(self.cfg.tracking.virtual_contact_force_cap),
            surface_velocity=self.virtual_surface_velocity,
            contact_model=str(self.cfg.tracking.contact_model),
            hunt_crossley_dissipation=self.virtual_contact_dissipation,
            power_law_exponent=self.virtual_contact_power_exponent,
            exponential_sharpness=float(self.cfg.tracking.exponential_sharpness),
            exponential_reference_force=float(self.cfg.tracking.power_law_reference_force),
            exponential_reference_penetration=float(self.cfg.tracking.power_law_reference_penetration),
            max_damping_multiplier=float(self.cfg.tracking.max_damping_multiplier),
            force_scale=float(self.cfg.tracking.virtual_contact_force_scale),
        )

        self.virtual_surface_distance[:] = distance
        self.virtual_contact_force[:] = contact_force
        self.current_contact[:] = distance.squeeze(-1) < 0.0
        if self.cfg.tracking.virtual_contact_torque_model == "orientation_spring":
            self.virtual_contact_torque[:] = orientation_spring_contact_torque(
                current_quat_world=self.fingertip_midpoint_quat,
                current_angvel_world=self.fingertip_midpoint_angvel,
                equilibrium_quat_world=self.virtual_rotational_equilibrium_quat,
                equilibrium_angvel_world=self.virtual_rotational_equilibrium_angvel,
                in_contact=self.current_contact,
                rotational_stiffness=float(
                    self.cfg.tracking.virtual_contact_rotational_stiffness
                ),
                rotational_damping=float(
                    self.cfg.tracking.virtual_contact_rotational_damping
                ),
                torque_scale=float(self.cfg.tracking.virtual_contact_torque_scale),
                torque_cap=float(self.cfg.tracking.virtual_contact_torque_cap),
            )
        elif self.cfg.tracking.virtual_contact_torque_model == "orientation_power_law":
            self.virtual_contact_torque[:] = orientation_power_law_contact_torque(
                current_quat_world=self.fingertip_midpoint_quat,
                current_angvel_world=self.fingertip_midpoint_angvel,
                equilibrium_quat_world=self.virtual_rotational_equilibrium_quat,
                equilibrium_angvel_world=self.virtual_rotational_equilibrium_angvel,
                in_contact=self.current_contact,
                rotational_coefficient=self.virtual_contact_rotational_coefficient,
                power_law_exponent=self.virtual_contact_rotational_power_exponent,
                rotational_damping=self.virtual_contact_rotational_damping,
                torque_scale=float(self.cfg.tracking.virtual_contact_torque_scale),
                torque_cap=float(self.cfg.tracking.virtual_contact_torque_cap),
            )
        else:
            self.virtual_contact_torque[:] = contact_coupled_torque(
                # Compare forces in physical units. At the demonstrated pose the
                # actual force includes this same post-model physical scale, so the
                # torque ratio remains one even when force is physically softened.
                target_force_world=(
                    self.virtual_target_force
                    * float(self.cfg.tracking.virtual_contact_force_scale)
                ),
                target_torque_world=self.virtual_target_torque,
                actual_force_world=self.virtual_contact_force,
                in_contact=self.current_contact,
                max_multiplier=float(self.cfg.tracking.max_damping_multiplier),
                torque_cap=float(self.cfg.tracking.virtual_contact_torque_cap),
                torque_scale=float(self.cfg.tracking.virtual_contact_torque_scale),
            )

    def _update_virtual_contact_sensor(self):
        """Sample and filter the virtual sensor once per physics substep.

        For Forge data, emulate ``get_link_incoming_joint_force``: negate the
        external world wrench and express it in the current rotating sensor axes.
        Legacy world-external datasets retain their prior world-axis feedback.
        The caller is ``_apply_action``, matching Forge's per-physics-substep
        force EMA independently of the policy decimation. The default alpha is
        adjusted so the 60 Hz filter has Forge's 120 Hz physical-time response.
        """
        virtual_wrench = self.virtual_contact_force
        if self.wrench_dim == 6:
            virtual_wrench = torch.cat(
                (self.virtual_contact_force, self.virtual_contact_torque),
                dim=-1,
            )
        if self.dataset_wrench_convention == "sensor_child_joint_reaction":
            sensor_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
            self.force_sensor_raw[:] = world_external_to_sensor_reaction(
                virtual_wrench, sensor_quat_w
            )
        else:
            self.force_sensor_raw[:] = virtual_wrench
        alpha = float(self.cfg.tracking.force_sensor_smoothing_factor)
        alpha = min(max(alpha, 0.0), 1.0)
        self.force_sensor_smooth[:] = (
            alpha * self.force_sensor_raw + (1.0 - alpha) * self.force_sensor_smooth
        )

    @staticmethod
    def _orthonormal_basis(axis: torch.Tensor):
        """Return two unit vectors spanning the plane orthogonal to `axis` (n, 3)."""
        helper = torch.zeros_like(axis)
        helper[:, 0] = 1.0
        # Swap the helper axis where `axis` is nearly parallel to x to avoid a zero cross product.
        near_x = axis[:, 0].abs() > 0.9
        helper[near_x, 0] = 0.0
        helper[near_x, 1] = 1.0
        perp1 = torch.linalg.cross(axis, helper)
        perp1 = perp1 / torch.clamp(torch.linalg.norm(perp1, dim=-1, keepdim=True), min=1.0e-6)
        perp2 = torch.linalg.cross(axis, perp1)
        return perp1, perp2

    def _future_wrench(self) -> torch.Tensor:
        """Reference-window target wrenches normalized by peak.

        Returns either 3 or 6 values per future step in the configured dataset
        convention. Force is normalized by force_mag_range[1]; full-wrench mode
        additionally normalizes torque by torque_mag_max. The window is strictly
        future-facing and starts at the next trajectory waypoint.
        """
        num_steps = self.cfg.tracking.num_future_steps
        base_idx = self._policy_step_to_reference_index(
            self.episode_length_buf
        ) + 1
        return torch.cat(
            [
                self._traj_wrench_at_index(base_idx + i) / self._wrench_scale
                for i in range(num_steps)
            ],
            dim=-1,
        )

    def _reference_accelerations_at_policy_step(
        self,
        policy_step: torch.Tensor,
        difference_interval: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current/previous Cartesian reference accelerations.

        Reference poses are sampled on the same interpolated policy clock as the
        active command. Clamping the first three historical indices to zero
        reproduces the environment's zero derivative initialization at reset.
        """

        interval = float(difference_interval)
        if interval <= 0.0:
            raise ValueError("difference_interval must be positive")
        poses = [
            self._traj_pose_at_policy_step((policy_step - offset).clamp(min=0))
            for offset in range(4)
        ]
        positions = [pose[0] for pose in poses]
        quaternions = [pose[1] for pose in poses]

        linear_velocity = (positions[0] - positions[1]) / interval
        previous_linear_velocity = (positions[1] - positions[2]) / interval
        previous_previous_linear_velocity = (positions[2] - positions[3]) / interval
        linear_acceleration = (linear_velocity - previous_linear_velocity) / interval
        previous_linear_acceleration = (
            previous_linear_velocity - previous_previous_linear_velocity
        ) / interval

        angular_velocity = math_utils.quat_box_minus(quaternions[0], quaternions[1]) / interval
        previous_angular_velocity = (
            math_utils.quat_box_minus(quaternions[1], quaternions[2]) / interval
        )
        previous_previous_angular_velocity = (
            math_utils.quat_box_minus(quaternions[2], quaternions[3]) / interval
        )
        angular_acceleration = (angular_velocity - previous_angular_velocity) / interval
        previous_angular_acceleration = (
            previous_angular_velocity - previous_previous_angular_velocity
        ) / interval
        return (
            linear_acceleration,
            angular_acceleration,
            previous_linear_acceleration,
            previous_angular_acceleration,
        )

    def _update_command(self):
        """Set the active interpolated pose and mode-specific wrench target."""
        self.command_pos[:], self.command_quat[:] = (
            self._traj_pose_at_policy_step(self.episode_length_buf)
        )
        if self.enable_force:
            if self.virtual_contact_enabled:
                self._set_virtual_contact_surface_at_policy_step(
                    self.episode_length_buf
                )
            else:
                idx = self._policy_step_to_reference_index(
                    self.episode_length_buf
                ).clamp(0, self._traj_len - 1)
                self.target_wrench[:] = self.traj_wrench_buf[
                    self._env_arange, idx
                ]
                self.current_contact[:] = self.traj_contact_buf[self._env_arange, idx]

    def _future_command_errors(self) -> torch.Tensor:
        """Errors to the strictly future reference-pose window.

        The window contains native 15 Hz waypoints and begins after the native
        segment containing the current policy step. Multiple async policy steps
        therefore see the same four native future samples.
        """
        num_steps = self.cfg.tracking.num_future_steps
        base_idx = self._policy_step_to_reference_index(
            self.episode_length_buf
        ) + 1

        errors = []
        for i in range(num_steps):
            target_pos, target_quat = self._traj_pose_at_index(base_idx + i)
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            errors.append(pos_error)
            errors.append(axis_angle_error)
        return torch.cat(errors, dim=-1)

    def _randomize_controller(self, env_ids: torch.Tensor):
        gains = self.default_gains[env_ids].clone()
        noise = torch.tensor(self.cfg.ctrl.task_prop_gains_noise_level, device=self.device).unsqueeze(0)
        scale_range = self.cfg.ctrl.task_prop_gains_randomization_scale_range
        mode = self.cfg.ctrl.task_prop_gains_randomization_mode
        if mode not in {"scalar", "per_axis"}:
            raise ValueError(
                "ctrl.task_prop_gains_randomization_mode must be 'scalar' or "
                f"'per_axis', got {mode!r}"
            )
        if mode == "scalar":
            if scale_range is None and not torch.allclose(noise, noise[:, :1].expand_as(noise)):
                raise ValueError(
                    "Scalar controller-gain randomization requires all entries in "
                    "ctrl.task_prop_gains_noise_level to be equal."
                )
        if scale_range is not None:
            if torch.any(noise > 0):
                raise ValueError(
                    "Set either ctrl.task_prop_gains_randomization_scale_range or "
                    "ctrl.task_prop_gains_noise_level, not both."
                )
            if len(scale_range) != 2:
                raise ValueError("ctrl.task_prop_gains_randomization_scale_range must be [low, high].")
            low, high = (float(value) for value in scale_range)
            if not 0.0 < low <= 1.0 <= high:
                raise ValueError(
                    "ctrl.task_prop_gains_randomization_scale_range must satisfy 0 < low <= 1 <= high."
                )
            sample_shape = (env_ids.numel(), 1) if mode == "scalar" else gains.shape
            multiplier = low + (high - low) * torch.rand(sample_shape, device=self.device)
            gains = gains * multiplier
        elif torch.any(noise > 0):
            if mode == "scalar":
                multiplier = 1.0 + (
                    2.0 * torch.rand((env_ids.numel(), 1), device=self.device) - 1.0
                ) * noise[:, :1]
                gains = gains * multiplier
            else:
                gains = gains * (1.0 + (2.0 * torch.rand_like(gains) - 1.0) * noise)
        self.task_prop_gains[env_ids] = gains
        deriv_gains = factory_utils.get_deriv_gains(gains)
        ratio_range = getattr(self.cfg.ctrl, "task_deriv_gains_damping_ratio_range", None)
        if ratio_range is not None:
            if len(ratio_range) != 2:
                raise ValueError("ctrl.task_deriv_gains_damping_ratio_range must be [low, high].")
            low, high = (float(value) for value in ratio_range)
            if not 0.0 < low <= high:
                raise ValueError(
                    "ctrl.task_deriv_gains_damping_ratio_range must satisfy 0 < low <= high."
                )
            zeta = low + (high - low) * torch.rand(
                (env_ids.numel(), 1), device=self.device
            )
        else:
            zeta = torch.full(
                (env_ids.numel(), 1),
                float(self.cfg.ctrl.control_gain_damping_ratio),
                device=self.device,
            )
        self.control_gain_damping_ratio[env_ids] = zeta
        deriv_gains = deriv_gains * zeta
        self.task_deriv_gains[env_ids] = deriv_gains

    def _sample_scaled_per_joint(
        self, num_samples: int, per_joint_range, scale_range, num_joints: int = 7
    ) -> torch.Tensor:
        """Sample per-joint bands multiplied by one shared scale per environment."""
        bounds = torch.as_tensor(per_joint_range, device=self.device, dtype=torch.float)
        if bounds.numel() == 2:
            bounds = bounds.reshape(1, 2).expand(num_joints, 2)
        elif tuple(bounds.shape) != (num_joints, 2):
            raise ValueError(
                f"Expected a [lo, hi] pair or a ({num_joints}, 2) per-joint table, "
                f"got shape {tuple(bounds.shape)}"
            )
        lower = bounds[:, 0].reshape(1, num_joints)
        upper = bounds[:, 1].reshape(1, num_joints)

        scale = torch.as_tensor(scale_range, device=self.device, dtype=torch.float).reshape(-1)
        if scale.numel() != 2:
            raise ValueError(f"Expected a [lo, hi] scale pair, got {tuple(scale.shape)}")
        level = scale[0] + (scale[1] - scale[0]) * torch.rand(
            (num_samples, 1), device=self.device
        )
        per_joint = lower + (upper - lower) * torch.rand(
            (num_samples, num_joints), device=self.device
        )
        return (per_joint * level).clamp(min=0.0)

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        env_ids_cpu = env_ids.cpu()
        masses = self.default_masses.clone()
        inertias = self.default_inertias.clone()
        coms = self.default_coms.clone()

        rand_cfg = self.cfg.randomization
        mass_device = masses.device
        default_masses = self.default_masses.to(mass_device)
        default_inertias = self.default_inertias.to(mass_device)

        if rand_cfg.enable_link_mass:
            lower, upper = rand_cfg.link_mass_scale_range
            scales = lower + (upper - lower) * torch.rand(
                (len(env_ids), self._robot.num_bodies), device=mass_device
            )
        else:
            scales = torch.ones((len(env_ids), self._robot.num_bodies), device=mass_device)

        masses[env_ids_cpu] = default_masses[env_ids_cpu] * scales
        inertias[env_ids_cpu] = default_inertias[env_ids_cpu] * scales.unsqueeze(-1)

        # The payload is merged into the payload body as a rigid point mass: combined mass, mass-weighted
        # CoM, and parallel-axis inertia update. The simulated dynamics are then exactly "nominal robot +
        # point mass", while its weight is applied separately as an external wrench (robot gravity is
        # disabled, see `_apply_external_wrenches`). The controller stays blind to all of this because it
        # receives the reconstructed nominal mass matrix (see `_compute_arm_mass_matrix`).
        if rand_cfg.enable_payload:
            payload_mass_lower, payload_mass_upper = rand_cfg.payload_mass_range
            payload_mass = payload_mass_lower + (payload_mass_upper - payload_mass_lower) * torch.rand(
                (len(env_ids), 1), device=self.device
            )
            payload_com_ranges = torch.tensor(rand_cfg.payload_com_range, device=self.device)
            payload_com = payload_com_ranges[:, 0] + (
                payload_com_ranges[:, 1] - payload_com_ranges[:, 0]
            ) * torch.rand((len(env_ids), 3), device=self.device)
            self.payload_mass[env_ids] = payload_mass
            self.payload_com[env_ids] = payload_com

            body_idx = self.payload_body_idx
            m_p = payload_mass.to(mass_device)
            p = payload_com.to(mass_device)
            m_b = masses[env_ids_cpu, body_idx].unsqueeze(-1)
            c_b = coms[env_ids_cpu, body_idx, 0:3]
            m_new = m_b + m_p
            # Negative payload represents over-compensation, but the merged body
            # must retain enough positive mass for a finite CoM and valid inertia.
            if bool((m_new <= 0.25 * m_b).any()):
                worst = int(torch.argmin(m_new.squeeze(-1)))
                raise ValueError(
                    "randomization.payload_mass_range floor is too negative for "
                    f"this robot: body {self.cfg.randomization.payload_body_name!r} "
                    f"has mass {float(m_b[worst]):.4f} kg and a sampled payload of "
                    f"{float(m_p[worst]):+.4f} kg leaves {float(m_new[worst]):.4f} kg. "
                    "Keep the floor above -0.75 * the body mass printed here."
                )
            c_new = (m_b * c_b + m_p * p) / m_new
            d_b = c_b - c_new
            d_p = p - c_new
            eye = torch.eye(3, device=mass_device).unsqueeze(0)
            inertia_b = inertias[env_ids_cpu, body_idx].view(-1, 3, 3)
            shift_b = m_b.unsqueeze(-1) * (
                (d_b * d_b).sum(-1)[:, None, None] * eye - d_b.unsqueeze(-1) * d_b.unsqueeze(-2)
            )
            shift_p = m_p.unsqueeze(-1) * (
                (d_p * d_p).sum(-1)[:, None, None] * eye - d_p.unsqueeze(-1) * d_p.unsqueeze(-2)
            )
            masses[env_ids_cpu, body_idx] = m_new.squeeze(-1)
            coms[env_ids_cpu, body_idx, 0:3] = c_new
            inertias[env_ids_cpu, body_idx] = (inertia_b + shift_b + shift_p).view(-1, 9)
            # PhysX jacobians are referenced at the current (merged) CoM; record the link-frame offset
            # back to the nominal CoM for the controller-side mass-matrix reconstruction.
            self.nominal_com_offsets[env_ids, body_idx - 1] = (c_b - c_new).to(self.device)
        else:
            self.payload_mass[env_ids] = 0.0
            self.payload_com[env_ids] = 0.0
            self.nominal_com_offsets[env_ids, self.payload_body_idx - 1] = 0.0
        # First moment is a sys-id residual, not part of payload domain
        # randomization. Never let a prior replay candidate leak across resets.
        self.payload_first_moment[env_ids] = 0.0

        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

        self._dynamics_write_sim_step = self._sim_step_counter
        if not self._randomized_mass_matrix_validated:
            self._mass_matrix_checks_pending += 1
            self._randomized_mass_matrix_validated = True

        self.link_mass_scales[env_ids] = scales.to(self.device)

        if rand_cfg.enable_joint_friction:
            nominal = getattr(rand_cfg, "joint_friction_nominal", None)
            if nominal is None:
                joint_friction = self._sample_scaled_per_joint(
                    len(env_ids),
                    rand_cfg.joint_friction_range,
                    getattr(rand_cfg, "joint_friction_scale_range", (1.0, 1.0)),
                )
            else:
                nominal_t = torch.as_tensor(
                    nominal, device=self.device, dtype=torch.float
                ).reshape(1, 7)
                lower, upper = getattr(rand_cfg, "joint_friction_scale_range", (0.0, 1.5))
                level = lower + (upper - lower) * torch.rand(
                    (len(env_ids), 1), device=self.device
                )
                jitter = float(getattr(rand_cfg, "joint_friction_jitter", 0.0))
                if jitter > 0.0:
                    level = level * (
                        1.0 - jitter
                        + 2.0 * jitter * torch.rand((len(env_ids), 7), device=self.device)
                    )
                joint_friction = (nominal_t * level).clamp(min=0.0)
        else:
            joint_friction = torch.zeros((len(env_ids), 7), device=self.device)
        self.joint_friction[env_ids] = joint_friction

        if getattr(rand_cfg, "enable_joint_dynamic_ratio", False):
            dynamic_ratio = self._sample_scaled_per_joint(
                len(env_ids),
                getattr(rand_cfg, "joint_dynamic_ratio_range", [[1.0, 1.0]] * 7),
                getattr(rand_cfg, "joint_dynamic_ratio_scale_range", (1.0, 1.0)),
            ).clamp(0.0, 1.0)
        else:
            dynamic_ratio = torch.ones((len(env_ids), 7), device=self.device)
        # Write dynamic zero first so PhysX's static >= dynamic invariant is true
        # throughout the update, including while static friction is being changed.
        self._robot.write_joint_dynamic_friction_coefficient_to_sim(
            0.0, joint_ids=self.arm_joint_ids, env_ids=env_ids
        )
        self._robot.write_joint_friction_coefficient_to_sim(
            joint_friction, joint_ids=self.arm_joint_ids, env_ids=env_ids
        )
        self._robot.write_joint_dynamic_friction_coefficient_to_sim(
            torch.minimum(dynamic_ratio * joint_friction, joint_friction),
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

        if getattr(rand_cfg, "enable_joint_viscous", False):
            joint_viscous = self._sample_scaled_per_joint(
                len(env_ids),
                getattr(rand_cfg, "joint_viscous_range", [[0.0, 0.0]] * 7),
                getattr(rand_cfg, "joint_viscous_scale_range", (1.0, 1.0)),
            )
        else:
            joint_viscous = torch.zeros((len(env_ids), 7), device=self.device)
        self._robot.write_joint_viscous_friction_coefficient_to_sim(
            joint_viscous, joint_ids=self.arm_joint_ids, env_ids=env_ids
        )

        if rand_cfg.enable_joint_armature:
            joint_armature = self._sample_scaled_per_joint(
                len(env_ids),
                rand_cfg.joint_armature_range,
                getattr(rand_cfg, "joint_armature_scale_range", (1.0, 1.0)),
            )
        else:
            joint_armature = torch.zeros((len(env_ids), 7), device=self.device)
        self.joint_armature[env_ids] = joint_armature
        self._robot.write_joint_armature_to_sim(
            joint_armature,
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

    def render(self, recompute: bool = False):
        """Render and overlay env-0 per-step tracking error on the top-right corner."""
        frame = super().render(recompute)
        if frame is None or self.render_mode != "rgb_array":
            return frame
        if self._vis_pos_error_norm is None or self._vis_rot_error_norm is None:
            return frame

        import cv2

        pos_err_mm = float(self._vis_pos_error_norm[0]) * 1000.0
        rot_err_deg = math.degrees(float(self._vis_rot_error_norm[0]))
        lines = [
            f"trans err: {pos_err_mm:6.1f} mm",
            f"rot err:   {rot_err_deg:6.2f} deg",
        ]

        # The annotator returns an RGBA buffer; draw on a contiguous RGB copy.
        frame = np.ascontiguousarray(frame[:, :, :3])
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        pad = 6
        line_h = 22
        text_w = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines)
        box_w = text_w + 2 * pad
        box_h = line_h * len(lines) + pad
        x0 = w - box_w - 10
        y0 = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0)
        for i, text in enumerate(lines):
            y = y0 + pad + (i + 1) * line_h - 6
            cv2.putText(frame, text, (x0 + pad, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        return frame

    def _set_debug_vis_impl(self, debug_vis: bool):
        pose_visible = bool(debug_vis and self.cfg.debug_vis)
        force_visible = bool(
            debug_vis
            and self.cfg.debug_vis_force
            and self.virtual_contact_enabled
        )

        if pose_visible:
            if not hasattr(self, "command_pose_visualizer"):
                frame_cfg = FRAME_MARKER_CFG.copy()
                frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
                frame_cfg.prim_path = "/Visuals/Command/pose"
                self.command_pose_visualizer = VisualizationMarkers(frame_cfg)

                # Current end-effector pose (smaller frame) so the gap to the
                # command frame shows the live tracking error.
                ee_cfg = FRAME_MARKER_CFG.copy()
                ee_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                ee_cfg.prim_path = "/Visuals/EndEffector/pose"
                self.ee_pose_visualizer = VisualizationMarkers(ee_cfg)

                path_cfg = SPHERE_MARKER_CFG.copy()
                path_cfg.markers["sphere"].radius = 0.004
                path_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                path_cfg.prim_path = "/Visuals/Command/traj_path"
                self.traj_path_visualizer = VisualizationMarkers(path_cfg)

                future_cfg = SPHERE_MARKER_CFG.copy()
                future_cfg.markers["sphere"].radius = 0.008
                future_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.4, 1.0)
                future_cfg.prim_path = "/Visuals/Command/future_targets"
                self.future_target_visualizer = VisualizationMarkers(future_cfg)
        if hasattr(self, "command_pose_visualizer"):
            self.command_pose_visualizer.set_visibility(pose_visible)
            self.ee_pose_visualizer.set_visibility(pose_visible)
            self.traj_path_visualizer.set_visibility(pose_visible)
            self.future_target_visualizer.set_visibility(pose_visible)

        if force_visible and not hasattr(self, "virtual_contact_plane_visualizer"):
            force_style = str(self.cfg.debug_vis_force_style)
            force_source = str(self.cfg.debug_vis_force_source)
            plane_cfg = CUBOID_MARKER_CFG.copy()
            plane_size = float(self.cfg.debug_vis_force_plane_size)
            plane_cfg.markers["cuboid"].size = (0.001, plane_size, plane_size)
            plane_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
            plane_cfg.markers["cuboid"].visual_material.opacity = 0.3
            plane_cfg.prim_path = "/Visuals/VirtualContact/plane"
            self.virtual_contact_plane_visualizer = VisualizationMarkers(plane_cfg)

            (
                self.virtual_contact_normal_shaft_visualizer,
                self.virtual_contact_normal_head_visualizer,
            ) = self._make_force_arrow_visualizers(
                "/Visuals/VirtualContact/plane_normal",
                {"normal": (1.0, 1.0, 0.0)},
            )

            if bool(self.cfg.debug_vis_force_ee_sphere):
                ee_sphere_cfg = SPHERE_MARKER_CFG.copy()
                ee_sphere_cfg.markers["sphere"].radius = float(
                    self.cfg.debug_vis_force_ee_sphere_radius
                )
                ee_sphere_cfg.markers["sphere"].visual_material.diffuse_color = (
                    1.0,
                    1.0,
                    1.0,
                )
                ee_sphere_cfg.markers["sphere"].visual_material.emissive_color = (
                    0.4,
                    0.4,
                    0.4,
                )
                ee_sphere_cfg.prim_path = "/Visuals/VirtualContact/end_effector"
                self.virtual_contact_ee_visualizer = VisualizationMarkers(ee_sphere_cfg)

            if force_style in ("vector", "both") and force_source in (
                "reference",
                "both",
            ):
                (
                    self.virtual_target_force_shaft_visualizer,
                    self.virtual_target_force_head_visualizer,
                ) = self._make_force_arrow_visualizers(
                    "/Visuals/VirtualContact/reference_force",
                    {"reference": (0.0, 0.0, 1.0)},
                )

            if force_style in ("vector", "both") and force_source in (
                "realized",
                "both",
            ):
                (
                    self.virtual_actual_force_shaft_visualizer,
                    self.virtual_actual_force_head_visualizer,
                ) = self._make_force_arrow_visualizers(
                    "/Visuals/VirtualContact/realized_force",
                    {"realized": (1.0, 0.0, 0.0)},
                )

            if force_style in ("components", "both"):
                component_colors = {
                    "force_x": (1.0, 0.0, 0.0),
                    "force_y": (0.0, 1.0, 0.0),
                    "force_z": (0.0, 0.0, 1.0),
                }
                if force_source in ("reference", "both"):
                    (
                        self.virtual_target_force_components_shaft_visualizer,
                        self.virtual_target_force_components_head_visualizer,
                    ) = self._make_force_arrow_visualizers(
                        "/Visuals/VirtualContact/reference_force_components",
                        component_colors,
                    )
                if force_source in ("realized", "both"):
                    (
                        self.virtual_actual_force_components_shaft_visualizer,
                        self.virtual_actual_force_components_head_visualizer,
                    ) = self._make_force_arrow_visualizers(
                        "/Visuals/VirtualContact/realized_force_components",
                        component_colors,
                    )

            print(
                "[FrankaRobustTrack] force debug visualization: "
                "cyan=virtual plane, yellow=outward normal, white=current EE, "
                f"style={force_style}, source={force_source}; "
                "components use red=Fx, green=Fy, blue=Fz in world axes"
            )
        if hasattr(self, "virtual_contact_plane_visualizer"):
            self.virtual_contact_plane_visualizer.set_visibility(force_visible)
            self.virtual_contact_normal_shaft_visualizer.set_visibility(force_visible)
            self.virtual_contact_normal_head_visualizer.set_visibility(force_visible)
        if hasattr(self, "virtual_contact_ee_visualizer"):
            self.virtual_contact_ee_visualizer.set_visibility(force_visible)
        for prefix in (
            "virtual_target_force",
            "virtual_actual_force",
            "virtual_target_force_components",
            "virtual_actual_force_components",
        ):
            for part in ("shaft", "head"):
                visualizer = getattr(self, f"{prefix}_{part}_visualizer", None)
                if visualizer is not None:
                    visualizer.set_visibility(force_visible)

    def _make_force_arrow_visualizers(
        self,
        prim_path: str,
        marker_colors: dict[str, tuple[float, float, float]],
    ) -> tuple[VisualizationMarkers, VisualizationMarkers]:
        """Create separate cylinder-shaft and cone-head marker sets."""
        shaft_markers = {}
        head_markers = {}
        for name, color in marker_colors.items():
            material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=tuple(0.25 * value for value in color),
                roughness=0.7,
            )
            shaft_markers[name] = sim_utils.CylinderCfg(
                radius=float(self.cfg.debug_vis_force_shaft_radius),
                height=1.0,
                axis="Z",
                visual_material=material,
            )
            head_markers[name] = sim_utils.ConeCfg(
                radius=float(self.cfg.debug_vis_force_head_radius),
                height=1.0,
                axis="Z",
                visual_material=material.copy(),
            )
        shaft_cfg = VisualizationMarkersCfg(
            prim_path=f"{prim_path}_shaft",
            markers=shaft_markers,
        )
        head_cfg = VisualizationMarkersCfg(
            prim_path=f"{prim_path}_head",
            markers=head_markers,
        )
        return VisualizationMarkers(shaft_cfg), VisualizationMarkers(head_cfg)

    def _force_debug_arrow_geometry(
        self,
        origin_world: torch.Tensor,
        force_world: torch.Tensor,
        *,
        length_scale: float | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return shaft/head positions, scales, and orientation for clear arrows."""
        magnitude = torch.linalg.norm(force_world, dim=-1)
        orientation = z_axis_to_vector_quat(force_world)
        direction = force_world / magnitude.clamp_min(1.0e-8).unsqueeze(-1)
        if length_scale is None:
            length_scale = float(self.cfg.debug_vis_force_scale)
        total_length = magnitude * float(length_scale)
        max_head_length = float(self.cfg.debug_vis_force_head_length)
        head_length = torch.minimum(
            torch.full_like(total_length, max_head_length),
            0.4 * total_length,
        )
        shaft_length = (total_length - head_length).clamp_min(0.0)
        shaft_position = origin_world + 0.5 * shaft_length.unsqueeze(-1) * direction
        head_position = origin_world + (
            shaft_length + 0.5 * head_length
        ).unsqueeze(-1) * direction
        shaft_scale = torch.ones_like(force_world)
        shaft_scale[:, 2] = shaft_length
        head_scale = torch.ones_like(force_world)
        head_scale[:, 2] = head_length
        visible = (magnitude > 1.0e-6).unsqueeze(-1)
        shaft_scale = torch.where(visible, shaft_scale, torch.zeros_like(shaft_scale))
        head_scale = torch.where(visible, head_scale, torch.zeros_like(head_scale))
        return (
            shaft_position,
            shaft_scale,
            head_position,
            head_scale,
            orientation,
        )

    def _visualize_force_arrow(
        self,
        shaft_visualizer: VisualizationMarkers,
        head_visualizer: VisualizationMarkers,
        origin_world: torch.Tensor,
        force_world: torch.Tensor,
        marker_indices: torch.Tensor | None = None,
        *,
        length_scale: float | None = None,
    ):
        """Render vectors as cylinder shafts capped by conical arrowheads."""
        (
            shaft_position,
            shaft_scale,
            head_position,
            head_scale,
            orientation,
        ) = self._force_debug_arrow_geometry(
            origin_world,
            force_world,
            length_scale=length_scale,
        )
        shaft_visualizer.visualize(
            shaft_position,
            orientation,
            shaft_scale,
            marker_indices,
        )
        head_visualizer.visualize(
            head_position,
            orientation,
            head_scale,
            marker_indices,
        )

    def _visualize_force_components(
        self,
        shaft_visualizer: VisualizationMarkers,
        head_visualizer: VisualizationMarkers,
        origin_world: torch.Tensor,
        force_world: torch.Tensor,
    ):
        """Render signed world Fx/Fy/Fz as red/green/blue component arrows."""
        component_vectors = world_force_components(force_world).reshape(-1, 3)
        component_origins = (
            origin_world.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        )
        marker_indices = (
            torch.arange(3, device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .reshape(-1)
        )
        self._visualize_force_arrow(
            shaft_visualizer,
            head_visualizer,
            component_origins,
            component_vectors,
            marker_indices,
        )

    def _debug_vis_callback(self, event):
        # This callback is driven by a timeline event and can fire during
        # shutdown, after the scene has already been torn down.
        if not hasattr(self, "scene"):
            return
        env_origins = self.scene.env_origins

        if self.cfg.debug_vis and hasattr(self, "command_pose_visualizer"):
            # Current reference pose (frame marker).
            self.command_pose_visualizer.visualize(self.command_pos + env_origins, self.command_quat)

            # Current end-effector pose (frame marker); offset from the command
            # frame is the live tracking error.
            self.ee_pose_visualizer.visualize(
                self.fingertip_midpoint_pos + env_origins,
                self.fingertip_midpoint_quat,
            )

            # Full discretized reference path over the episode (strided waypoints).
            num_samples = min(self.cfg.debug_vis_path_samples, self._traj_len)
            sample_idx = torch.linspace(
                0, self._traj_len - 1, num_samples, device=self.device
            ).long()
            path_points = []
            for idx in sample_idx:
                path_points.append(self.traj_pos_buf[:, idx] + env_origins)
            self.traj_path_visualizer.visualize(torch.cat(path_points, dim=0))

            # Lookahead targets fed to the policy.
            base_idx = self._policy_step_to_reference_index(
                self.episode_length_buf
            ) + 1
            future_points = []
            for i in range(self.cfg.tracking.num_future_steps):
                pos, _ = self._traj_pose_at_index(base_idx + i)
                future_points.append(pos + env_origins)
            self.future_target_visualizer.visualize(torch.cat(future_points, dim=0))

        if (
            self.cfg.debug_vis_force
            and self.virtual_contact_enabled
            and hasattr(self, "virtual_contact_plane_visualizer")
        ):
            # The plane cuboid's thin local +x axis and the green arrow both
            # align with the exact outward normal used by the contact equation.
            plane_orientation = x_axis_to_vector_quat(self.virtual_surface_normal)
            self.virtual_contact_plane_visualizer.visualize(
                self.virtual_surface_point + env_origins,
                plane_orientation,
            )
            self._visualize_force_arrow(
                self.virtual_contact_normal_shaft_visualizer,
                self.virtual_contact_normal_head_visualizer,
                self.virtual_surface_point + env_origins,
                self.virtual_surface_normal,
                length_scale=float(self.cfg.debug_vis_force_normal_length),
            )
            if hasattr(self, "virtual_contact_ee_visualizer"):
                self.virtual_contact_ee_visualizer.visualize(
                    self.fingertip_midpoint_pos + env_origins
                )

            # The target marker starts at the demonstrated pose associated with
            # this plane. The realized marker starts where its force acts.
            reference_pos = self.command_pos
            if hasattr(self, "virtual_target_force_shaft_visualizer"):
                self._visualize_force_arrow(
                    self.virtual_target_force_shaft_visualizer,
                    self.virtual_target_force_head_visualizer,
                    reference_pos + env_origins,
                    self.virtual_target_force,
                )
            if hasattr(self, "virtual_actual_force_shaft_visualizer"):
                self._visualize_force_arrow(
                    self.virtual_actual_force_shaft_visualizer,
                    self.virtual_actual_force_head_visualizer,
                    self.fingertip_midpoint_pos + env_origins,
                    self.virtual_contact_force,
                )
            if hasattr(self, "virtual_target_force_components_shaft_visualizer"):
                self._visualize_force_components(
                    self.virtual_target_force_components_shaft_visualizer,
                    self.virtual_target_force_components_head_visualizer,
                    reference_pos + env_origins,
                    self.virtual_target_force,
                )
            if hasattr(self, "virtual_actual_force_components_shaft_visualizer"):
                self._visualize_force_components(
                    self.virtual_actual_force_components_shaft_visualizer,
                    self.virtual_actual_force_components_head_visualizer,
                    self.fingertip_midpoint_pos + env_origins,
                    self.virtual_contact_force,
                )

    def _command_errors(self):
        pos_error, rot_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=self.command_pos,
            ctrl_target_fingertip_midpoint_quat=self.command_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        return pos_error, rot_error

    def _accumulate_contact_split_metrics(
        self,
        pos_error_norm: torch.Tensor,
        rot_error_norm: torch.Tensor,
        successes: torch.Tensor,
        env_mask: torch.Tensor | None = None,
    ):
        """Accumulate tracking error/success split by free-space vs contact steps.

        `self.current_contact` (num_envs,) flags which envs have a contact band
        active this step. Per-env errors/successes are summed into free and contact
        accumulators with their own counts, so the logged scalar is the mean over
        all (env, step) samples of that category in the running window.
        """
        if env_mask is not None:
            pos_error_norm = pos_error_norm[env_mask]
            rot_error_norm = rot_error_norm[env_mask]
            successes = successes[env_mask]
            contact = self.current_contact[env_mask]
        else:
            contact = self.current_contact
        free = ~contact
        succ_f = successes.float()

        self._track_free_pos_sum += pos_error_norm[free].sum()
        self._track_free_rot_sum += rot_error_norm[free].sum()
        self._track_free_succ_sum += succ_f[free].sum()
        self._track_free_count += free.sum()
        self._track_contact_pos_sum += pos_error_norm[contact].sum()
        self._track_contact_rot_sum += rot_error_norm[contact].sum()
        self._track_contact_succ_sum += succ_f[contact].sum()
        self._track_contact_count += contact.sum()

        free_n = self._track_free_count.clamp(min=1.0)
        contact_n = self._track_contact_count.clamp(min=1.0)
        self.extras["tracking_pos_error_free"] = self._track_free_pos_sum / free_n
        self.extras["tracking_rot_error_free"] = self._track_free_rot_sum / free_n
        self.extras["success_free"] = self._track_free_succ_sum / free_n
        self.extras["tracking_pos_error_contact"] = self._track_contact_pos_sum / contact_n
        self.extras["tracking_rot_error_contact"] = self._track_contact_rot_sum / contact_n
        self.extras["success_contact"] = self._track_contact_succ_sum / contact_n
        self.extras["contact_fraction"] = contact.float().mean()

    def _accumulate_perstep_error(
        self,
        pos_error_norm: torch.Tensor,
        rot_error_norm: torch.Tensor,
        env_mask: torch.Tensor | None = None,
    ):
        """Bin tracking error by native reference index and log it periodically."""
        if self._perstep_log_every <= 0:
            return
        step_idx = self._policy_step_to_reference_index(
            self.episode_length_buf
        ).clamp(0, self._perstep_num_bins - 1)
        if env_mask is not None:
            step_idx = step_idx[env_mask]
            pos_error_norm = pos_error_norm[env_mask]
            rot_error_norm = rot_error_norm[env_mask]
        self._perstep_pos_sum.index_add_(0, step_idx, pos_error_norm)
        self._perstep_pos_sqsum.index_add_(0, step_idx, pos_error_norm.square())
        self._perstep_rot_sum.index_add_(0, step_idx, rot_error_norm)
        self._perstep_rot_sqsum.index_add_(0, step_idx, rot_error_norm.square())
        self._perstep_count.index_add_(0, step_idx, torch.ones_like(pos_error_norm))
        self._perstep_step_counter += 1
        if self._perstep_step_counter >= self._perstep_log_every:
            self._log_perstep_error_profile()
            self._reset_perstep_error()

    def _reset_perstep_error(self):
        self._perstep_pos_sum.zero_()
        self._perstep_pos_sqsum.zero_()
        self._perstep_rot_sum.zero_()
        self._perstep_rot_sqsum.zero_()
        self._perstep_count.zero_()
        self._perstep_step_counter = 0

    def _log_perstep_error_profile(self):
        """Log a mean±std line plot of the per-episode-step tracking error to wandb."""
        import wandb

        if wandb.run is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = self._perstep_count > 0
        if not torch.any(valid):
            return
        count_safe = self._perstep_count.clamp(min=1.0)
        pos_mean = self._perstep_pos_sum / count_safe
        pos_std = (self._perstep_pos_sqsum / count_safe - pos_mean.square()).clamp(min=0.0).sqrt()
        rot_mean = self._perstep_rot_sum / count_safe
        rot_std = (self._perstep_rot_sqsum / count_safe - rot_mean.square()).clamp(min=0.0).sqrt()

        steps = torch.arange(self._perstep_num_bins, device=self.device)[valid]
        time_s = (steps.float() * self.reference_dt).cpu().numpy()
        pos_mean_np = pos_mean[valid].cpu().numpy()
        pos_std_np = pos_std[valid].cpu().numpy()
        rot_mean_np = rot_mean[valid].cpu().numpy()
        rot_std_np = rot_std[valid].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(time_s, pos_mean_np, color="C0", label="mean")
        axes[0].fill_between(
            time_s, pos_mean_np - pos_std_np, pos_mean_np + pos_std_np, color="C0", alpha=0.25, label="±1 std"
        )
        axes[0].set_xlabel("time in episode (s)")
        axes[0].set_ylabel("position error (m)")
        axes[0].set_title("Per-step position tracking error")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_s, rot_mean_np, color="C1", label="mean")
        axes[1].fill_between(
            time_s, rot_mean_np - rot_std_np, rot_mean_np + rot_std_np, color="C1", alpha=0.25, label="±1 std"
        )
        axes[1].set_xlabel("time in episode (s)")
        axes[1].set_ylabel("orientation error (rad)")
        axes[1].set_title("Per-step orientation tracking error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        wandb.log({"tracking/perstep_error_profile": wandb.Image(fig)})
        plt.close(fig)

    def _accumulate_gain_stats(self, env_mask: torch.Tensor | None = None):
        """Accumulate per-step policy-scheduled PD gains and log stats periodically."""
        if not self.cfg.ctrl.control_gains or self._perstep_log_every <= 0:
            return
        gains = self.task_prop_gains.detach()
        contact = self.current_contact
        if env_mask is not None:
            gains = gains[env_mask]
            contact = contact[env_mask]
        self._gain_sum += gains.sum(dim=0)
        self._gain_sqsum += gains.square().sum(dim=0)
        self._gain_count += gains.shape[0]

        if self.enable_force:
            free = ~contact
            self._gain_free_sum += gains[free].sum(dim=0)
            self._gain_free_sqsum += gains[free].square().sum(dim=0)
            self._gain_free_count += free.sum()
            self._gain_contact_sum += gains[contact].sum(dim=0)
            self._gain_contact_sqsum += gains[contact].square().sum(dim=0)
            self._gain_contact_count += contact.sum()

        gain_min = self.gain_min[0]
        gain_max = self.gain_max[0]
        span = (gain_max - gain_min).clamp(min=1.0e-6)
        normalized = (gains - gain_min) / span
        bin_idx = (normalized * self._gain_hist_num_bins).long().clamp(0, self._gain_hist_num_bins - 1)
        for d in range(self._gain_stats_num_dims):
            self._gain_hist[d].index_add_(0, bin_idx[:, d], torch.ones_like(bin_idx[:, d], dtype=torch.float))

        self._gain_step_counter += 1
        if self._gain_step_counter >= self._perstep_log_every:
            self._log_gain_stats()
            self._reset_gain_stats()

    def _reset_gain_stats(self):
        self._gain_sum.zero_()
        self._gain_sqsum.zero_()
        self._gain_count.zero_()
        self._gain_hist.zero_()
        self._gain_step_counter = 0
        for buf in (
            self._gain_free_sum, self._gain_free_sqsum, self._gain_free_count,
            self._gain_contact_sum, self._gain_contact_sqsum, self._gain_contact_count,
        ):
            buf.zero_()

    def _log_gain_stats(self):
        """Log mean/std scalars and a per-dim histogram of the scheduled PD gains to wandb."""
        import wandb

        if wandb.run is None or float(self._gain_count) <= 0:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean = self._gain_sum / self._gain_count
        std = (self._gain_sqsum / self._gain_count - mean.square()).clamp(min=0.0).sqrt()

        # Axis labels: first 3 dims are translational stiffness, last 3 rotational.
        axis_names = ["x", "y", "z", "rx", "ry", "rz"]
        log_dict = {}
        for d, name in enumerate(axis_names):
            log_dict[f"gains/mean_{name}"] = float(mean[d])
            log_dict[f"gains/std_{name}"] = float(std[d])

        # Free-space vs contact split of the scheduled-gain mean/std.
        if self.enable_force:
            for phase, gsum, gsqsum, gcount in (
                ("free", self._gain_free_sum, self._gain_free_sqsum, self._gain_free_count),
                ("contact", self._gain_contact_sum, self._gain_contact_sqsum, self._gain_contact_count),
            ):
                if float(gcount) <= 0:
                    continue
                phase_mean = gsum / gcount
                phase_std = (gsqsum / gcount - phase_mean.square()).clamp(min=0.0).sqrt()
                for d, name in enumerate(axis_names):
                    log_dict[f"gains/mean_{name}_{phase}"] = float(phase_mean[d])
                    log_dict[f"gains/std_{name}_{phase}"] = float(phase_std[d])

        gain_min = self.gain_min[0].cpu().numpy()
        gain_max = self.gain_max[0].cpu().numpy()
        hist = self._gain_hist.cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for d, name in enumerate(axis_names):
            edges = torch.linspace(0.0, 1.0, self._gain_hist_num_bins + 1).cpu().numpy()
            centers = 0.5 * (edges[:-1] + edges[1:])
            centers = gain_min[d] + centers * (gain_max[d] - gain_min[d])
            width = (gain_max[d] - gain_min[d]) / self._gain_hist_num_bins
            axes[d].bar(centers, hist[d], width=width, color="C0", alpha=0.8)
            axes[d].axvline(float(mean[d]), color="C3", linestyle="--", label=f"mean={float(mean[d]):.1f}")
            axes[d].set_xlim(gain_min[d], gain_max[d])
            axes[d].set_xlabel(f"prop gain [{name}]")
            axes[d].set_ylabel("count")
            axes[d].set_title(f"gain[{name}] (std={float(std[d]):.1f})")
            axes[d].legend()
            axes[d].grid(True, alpha=0.3)

        fig.tight_layout()
        log_dict["gains/histogram"] = wandb.Image(fig)
        wandb.log(log_dict)
        plt.close(fig)

    def _joint_limit_penalty(self):
        soft_limit = 0.95 * math.pi
        excess = torch.relu(torch.abs(self.joint_pos[:, 0:7]) - soft_limit)
        return torch.sum(excess, dim=-1)

    def _log_reset_metrics(self, env_ids: torch.Tensor):
        self.extras["log"] = {}
        if not hasattr(self, "_episode_sums"):
            return
        for key, episodic_sum in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(episodic_sum[env_ids]) / self.max_episode_length_s
            episodic_sum[env_ids] = 0.0
        if self._adaptive_phase_enabled:
            failure_rate = (
                self._adaptive_phase_failure_counts
                / self._adaptive_phase_visit_counts
            )
            self.extras["log"]["AdaptivePhase/failure_rate_mean"] = (
                failure_rate.mean()
            )
            self.extras["log"]["AdaptivePhase/failure_rate_max"] = (
                failure_rate.max()
            )
            if self.cfg.tracking.mode == "dataset":
                self.extras["log"]["AdaptivePhase/start_time_mean_s"] = (
                    self.traj_ds_start[env_ids].float().mean()
                    * self.reference_dt
                )
                probabilities = self._adaptive_dataset_probabilities()
            else:
                self.extras["log"]["AdaptivePhase/start_time_mean_s"] = (
                    self.traj_start_time[env_ids].mean()
                )
                start_time_lo, start_time_hi = (
                    self.cfg.tracking.circle_start_time_range
                )
                probabilities = None
                if start_time_lo < start_time_hi:
                    probabilities = self._adaptive_phase_probabilities(
                        start_time_lo=start_time_lo,
                        start_time_hi=start_time_hi,
                    )
            if probabilities is not None:
                entropy = -(
                    probabilities
                    * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
                ).sum()
                self.extras["log"]["AdaptivePhase/sampling_entropy"] = entropy
                self.extras["log"]["AdaptivePhase/max_probability"] = (
                    probabilities.max()
                )
        if self._collect_ee_derivative_analytics:
            diagnostics = self._episode_ee_derivative_diagnostics
            duration = float(self.max_episode_length_s)
            self.extras["log"]["EE_Diagnostics/linear_accel_mean"] = (
                diagnostics["accel_sum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/linear_accel_rms"] = torch.sqrt(
                diagnostics["accel_sqsum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/linear_accel_max"] = diagnostics[
                "accel_max"
            ][env_ids].max()
            self.extras["log"]["EE_Diagnostics/linear_jerk_mean"] = (
                diagnostics["jerk_sum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/linear_jerk_rms"] = torch.sqrt(
                diagnostics["jerk_sqsum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/linear_jerk_max"] = diagnostics[
                "jerk_max"
            ][env_ids].max()
            self.extras["log"]["EE_Diagnostics/angular_accel_mean"] = (
                diagnostics["angular_accel_sum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/angular_accel_rms"] = torch.sqrt(
                diagnostics["angular_accel_sqsum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/angular_accel_max"] = diagnostics[
                "angular_accel_max"
            ][env_ids].max()
            self.extras["log"]["EE_Diagnostics/angular_jerk_mean"] = (
                diagnostics["angular_jerk_sum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/angular_jerk_rms"] = torch.sqrt(
                diagnostics["angular_jerk_sqsum"][env_ids].mean() / duration
            )
            self.extras["log"]["EE_Diagnostics/angular_jerk_max"] = diagnostics[
                "angular_jerk_max"
            ][env_ids].max()
            for log_name, prefix in (
                ("residual_linear_accel", "residual_linear_accel"),
                ("residual_linear_jerk", "residual_linear_jerk"),
                ("residual_angular_accel", "residual_angular_accel"),
                ("residual_angular_jerk", "residual_angular_jerk"),
            ):
                self.extras["log"][f"EE_Diagnostics/{log_name}_mean"] = (
                    diagnostics[f"{prefix}_sum"][env_ids].mean() / duration
                )
                self.extras["log"][f"EE_Diagnostics/{log_name}_rms"] = torch.sqrt(
                    diagnostics[f"{prefix}_sqsum"][env_ids].mean() / duration
                )
                self.extras["log"][f"EE_Diagnostics/{log_name}_max"] = diagnostics[
                    f"{prefix}_max"
                ][env_ids].max()
            for value in diagnostics.values():
                value[env_ids] = 0.0
        self.extras["log"]["Dynamics/payload_mass"] = self.payload_mass[env_ids].mean()
        self.extras["log"]["Dynamics/joint_friction"] = self.joint_friction[env_ids].mean()
        self.extras["log"]["Dynamics/joint_armature"] = self.joint_armature[env_ids].mean()
        if self._action_latency_buf is not None:
            self.extras["log"]["Dynamics/action_latency_steps"] = (
                self._action_latency_steps[env_ids].float().mean()
            )
        self.extras["log"]["Dynamics/target_update_delay_substeps"] = (
            self._target_update_delay_steps[env_ids].float().mean()
        )
        if self.virtual_contact_enabled:
            self.extras["log"]["VirtualContact/surface_stiffness"] = self.virtual_surface_stiffness[
                env_ids
            ].mean()
            self.extras["log"]["VirtualContact/surface_damping"] = self.virtual_surface_damping[
                env_ids
            ].mean()
