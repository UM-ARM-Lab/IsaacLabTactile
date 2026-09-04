# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import axis_angle_from_quat, quat_apply

from isaaclab_tasks.direct.factory import factory_control, factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import forge_utils
from .forge_env_cfg import ForgeEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor information.
        self.force_sensor_body_name = (
            "force_sensor"
            if "force_sensor" in self._robot.body_names
            else "panda_fingertip_centered"
        )
        self.force_sensor_body_idx = self._robot.body_names.index(
            self.force_sensor_body_name
        )
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.ep_max_ee_speed = torch.zeros((self.num_envs,), device=self.device)

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

        # Per-step action packets can carry action_rep and optional compliance
        # gains without mutating env hooks from the caller side.
        self.current_action_rep = str(getattr(self.cfg.ctrl, "action_rep", "rel_ee_pose"))
        self.delta_target_mode = str(
            getattr(self.cfg.ctrl, "delta_target_mode", "per_physics_step")
        )
        if self.delta_target_mode not in ("per_physics_step", "per_control_step"):
            raise ValueError(
                "ctrl.delta_target_mode must be 'per_physics_step' or "
                f"'per_control_step', got {self.delta_target_mode!r}."
            )
        self._delta_action_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._delta_action_target_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        self.delta_rot = torch.zeros((self.num_envs, 3), device=self.device)
        self.packet_stiffness_override = None
        self.packet_deriv_override = None
        self.packet_clip_pose_target = True
        self.last_ctrl_target_fingertip_midpoint_pos = None
        self.last_ctrl_target_fingertip_midpoint_quat = None
        # Matrix-valued gain channel for directional (anisotropic) compliance.
        # When set (num_envs, 6, 6), generate_ctrl_signals uses these instead of
        # the per-axis task_prop_gains / task_deriv_gains vectors.
        self.task_prop_gains_matrix = None
        self.task_deriv_gains_matrix = None

        # Cache the nominal (unrandomized) arm inertial parameters for the
        # controller-side mass-matrix reconstruction used when
        # `ctrl.use_gt_mass_matrix` is False. See `_compute_arm_mass_matrix`.
        if not self.cfg.ctrl.use_gt_mass_matrix:
            self._cache_nominal_arm_dynamics()

    def _cache_nominal_arm_dynamics(self):
        """Snapshot the default per-link inertial parameters (base body dropped).

        These are the config-default arm inertials read before any dynamics
        randomization, mirroring `FrankaRobustTrackEnv._cache_default_dynamics`.
        """
        num_links = self._robot.num_bodies - 1
        default_masses = self._robot.root_physx_view.get_masses().clone()
        default_inertias = self._robot.root_physx_view.get_inertias().clone()
        self.nominal_link_masses = default_masses[:, 1:].to(self.device)
        self.nominal_link_inertias = default_inertias[:, 1:].to(self.device).view(
            self.num_envs, num_links, 3, 3
        )
        # No payload merge into a robot link in forge, so the reconstruction
        # references each link's current CoM directly (zero offset).
        self.nominal_com_offsets = torch.zeros((self.num_envs, num_links, 3), device=self.device)
        # Nominal reflected rotor inertia the controller assumes (blind to any
        # per-env armature randomization written to PhysX).
        self.arm_armature = self._robot.root_physx_view.get_dof_armatures().to(self.device)[:, 0:7]

    def _compute_arm_mass_matrix(
        self,
        jacobians: torch.Tensor,
        link_masses: torch.Tensor,
        link_inertias: torch.Tensor,
        com_offsets: torch.Tensor,
        armature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rebuild the 7x7 arm mass matrix from per-link jacobians and inertials.

        M(q) = Σ_l m_l J_vc^T J_vc + J_w^T (R I R^T) J_w + diag(armature). PhysX
        link jacobians are referenced at each link's current CoM (world axes) and
        inertias are link-frame tensors about the CoM. Evaluated with the cached
        nominal parameters this yields the unrandomized-robot mass matrix,
        matching `FrankaRobustTrackEnv._compute_arm_mass_matrix`.
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

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)

        kinematics_mode = str(
            getattr(self.cfg.ctrl, "tool_kinematics_mode", "finger_average")
        )
        if kinematics_mode == "tool_body":
            body_pos_w = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
            body_com_pos_w = self._robot.data.body_com_pos_w[
                :, self.fingertip_body_idx
            ]
            body_com_linvel_w = self._robot.data.body_com_lin_vel_w[
                :, self.fingertip_body_idx
            ]
            body_angvel_w = self._robot.data.body_com_ang_vel_w[
                :, self.fingertip_body_idx
            ]
            com_to_tool_w = body_pos_w - body_com_pos_w
            self.fingertip_midpoint_linvel = (
                body_com_linvel_w
                + torch.linalg.cross(
                    body_angvel_w, com_to_tool_w, dim=-1
                )
            )
            self.fingertip_midpoint_angvel = body_angvel_w
            jacobians = self._robot.root_physx_view.get_jacobians()
            tool_jacobian = jacobians[
                :, self.fingertip_body_idx - 1, 0:6, 0:7
            ].clone()
            angular_cols = tool_jacobian[:, 3:6, :].transpose(1, 2)
            linear_shift = torch.linalg.cross(
                angular_cols,
                com_to_tool_w.unsqueeze(1).expand_as(angular_cols),
                dim=-1,
            ).transpose(1, 2)
            tool_jacobian[:, 0:3, :] += linear_shift
            self.fingertip_midpoint_jacobian = tool_jacobian
        elif kinematics_mode != "finger_average":
            raise ValueError(
                "ctrl.tool_kinematics_mode must be 'finger_average' or "
                f"'tool_body', got {kinematics_mode!r}"
            )

        # Optionally replace PhysX's ground-truth mass matrix (set by the factory
        # base) with the nominal reconstruction so the OSC task-space inertia
        # matches the franka_robust_track tracker trained with
        # use_gt_mass_matrix=False.
        if not self.cfg.ctrl.use_gt_mass_matrix:
            jacobians = self._robot.root_physx_view.get_jacobians()
            self.arm_mass_matrix = self._compute_arm_mass_matrix(
                jacobians,
                self.nominal_link_masses,
                self.nominal_link_inertias,
                self.nominal_com_offsets,
            )

        # Add noise to fingertip position.
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise
        # Add isotropic quaternion noise.  The legacy yaw-only policy projects
        # the result back onto the top-down quaternion manifold; a full-rotation
        # policy must retain all four components so it can observe wrist tilt.
        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True).clamp_min(1.0e-8)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        if not self.cfg.ctrl.use_full_rotation:
            self.noisy_fingertip_quat[:, [0, 3]] = 0.0
            self.noisy_fingertip_quat /= torch.linalg.norm(
                self.noisy_fingertip_quat, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.where(rot_diff_quat[:, :1] < 0.0, -1.0, 1.0)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        if not self.cfg.ctrl.use_full_rotation:
            self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # IMPORTANT: PhysX reports `get_link_incoming_joint_force()` in the
        # incoming joint's child frame, not in world axes.  The historical
        # `force_sensor_world*` names below are therefore misleading.
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]

        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        # NOTE: this legacy call treats the child-frame wrench above as if its
        # source frame had identity/world orientation.  Consequently
        # `force_sensor_smooth` is not, in general, a wrench expressed about the
        # fixed-object observation frame.  A correct conversion must use the
        # sensor child-joint pose and a full wrench (dual-adjoint) transform.
        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _parse_action_packet(self, action_or_packet):
        """Return (action, action_rep, compliance) for tensor or dict step input."""
        if not isinstance(action_or_packet, dict):
            return action_or_packet, str(getattr(self.cfg.ctrl, "action_rep", "rel_ee_pose")), {}
        if "action" not in action_or_packet:
            raise KeyError("Action packet must contain an 'action' tensor.")
        action = action_or_packet["action"]
        action_rep = str(action_or_packet.get("action_rep", getattr(self.cfg.ctrl, "action_rep", "rel_ee_pose")))
        compliance = action_or_packet.get("compliance", None) or {}
        return action, action_rep, compliance

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing, or unpack an action packet."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        action, action_rep, compliance = self._parse_action_packet(action)
        if action_rep not in (
            "rel_ee_pose",
            "abs_ee_pose",
            "delta_ee_pose",
            "delta_joint_pos",
        ):
            raise ValueError(f"Unsupported Forge action_rep={action_rep!r}")

        self.current_action_rep = action_rep
        self.packet_stiffness_override = compliance.get("stiffness", None)
        self.packet_deriv_override = compliance.get("damping", None)
        self.packet_clip_pose_target = bool(compliance.get("clip_pose_target", True))

        action = action.clone().to(self.device)
        if action_rep == "abs_ee_pose":
            if action.shape[-1] < 7:
                raise ValueError(f"action_rep='abs_ee_pose' requires at least 7 dims, got {tuple(action.shape)}")
            self.actions[:, 0:7] = action[:, 0:7]
            return

        if action.shape[-1] < self.actions.shape[-1]:
            pad = torch.zeros(
                action.shape[0],
                self.actions.shape[-1] - action.shape[-1],
                device=self.device,
                dtype=action.dtype,
            )
            action = torch.cat((action, pad), dim=-1)
        elif action.shape[-1] > self.actions.shape[-1]:
            action = action[:, : self.actions.shape[-1]]
        self.actions = self.ema_factor * action + (1 - self.ema_factor) * self.actions
        if action_rep == "delta_ee_pose" and self.delta_target_mode == "per_control_step":
            # Snapshot current_pose + delta once at the control-step boundary.
            # `_apply_action` is called once per decimation substep, so keeping
            # this absolute target fixed prevents repeated delta accumulation.
            target_pos, target_quat = self._get_delta_ee_pose_target()
            self._delta_action_target_pos.copy_(target_pos)
            self._delta_action_target_quat.copy_(target_quat)

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        if not self.cfg.ctrl.use_full_rotation:
            prev_actions[:, 3:5] = 0.0

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": self.force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })
        self._update_obs_history(obs_dict)
        if self.obs_history_length > 0:
            history_tensors = [self.obs_history_buffers[obs_name].reshape(self.num_envs, -1) for obs_name in self.cfg.obs_order + ["prev_actions"]]
            obs_tensors = torch.cat(history_tensors, dim=1)
        else:
            # No history: use current observations only
            obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Compliance-eval: optionally override the task-space stiffness for this
        # step. Applies to both the standard and pose-target paths below.
        # Reset the matrix-gain channel each step; only the directional override
        # below repopulates it. `task_prop_gains` itself stays a (num_envs, 6)
        # vector so the observation builder (which includes task_prop_gains in
        # state_order) keeps working.
        self.task_prop_gains_matrix = None
        self.task_deriv_gains_matrix = None
        override = self.packet_stiffness_override
        deriv_override = self.packet_deriv_override
        if override is not None:
            if override.dim() == 3:
                # Full (num_envs, 6, 6) anisotropic stiffness: route it (and its
                # matching damping) to the controller via the matrix channel,
                # leaving the per-axis task_prop_gains untouched for observations.
                self.task_prop_gains_matrix = override
                self.task_deriv_gains_matrix = deriv_override
            else:
                self.task_prop_gains = override
                if deriv_override is not None:
                    self.task_deriv_gains = deriv_override
                else:
                    # Per-axis prop -> critical damping (legacy isotropic path).
                    self.task_deriv_gains = factory_utils.get_deriv_gains(override)

        if self.current_action_rep == "abs_ee_pose":
            self._apply_abs_ee_pose_action(self.actions[:, 0:7], self.packet_clip_pose_target)
            return
        if self.current_action_rep == "delta_ee_pose":
            self._apply_delta_ee_pose_action()
            return
        if self.current_action_rep == "delta_joint_pos":
            self._apply_delta_joint_pos_action()
            return

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        # fixed_pos_action_frame = self.fixed_pos_obs_frame
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        if not self.cfg.ctrl.use_full_rotation:
            rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame. [-1, 1] -> [-180, 90]
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )
        # Assume bolt is point upright, so flip to get ee pose (topdown)
        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)
        self._update_rotation_target_error(ctrl_target_fingertip_preclipped_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.cfg.ctrl.gripper_dof_pos,
        )

    def _get_delta_ee_pose_target(self):
        """Map the normalized delta action to a target anchored at the current EE."""
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

    def _update_rotation_target_error(self, target_quat: torch.Tensor) -> None:
        """Cache the shortest 3-axis target error for reward computation."""
        quat_error = torch_utils.quat_mul(
            target_quat, torch_utils.quat_conjugate(self.fingertip_midpoint_quat)
        )
        quat_sign = torch.where(quat_error[:, :1] < 0.0, -1.0, 1.0)
        self.delta_rot = axis_angle_from_quat(quat_error * quat_sign)

    def _apply_delta_ee_pose_action(self):
        """Apply a current-EE delta using the configured target update rate."""
        pos_actions = self.actions[:, 0:3] * self.pos_threshold
        rot_actions = self.actions[:, 3:6] * self.rot_threshold
        self.delta_pos = pos_actions
        self.delta_yaw = torch.norm(rot_actions, p=2, dim=-1)

        if self.delta_target_mode == "per_control_step":
            target_pos = self._delta_action_target_pos
            target_quat = self._delta_action_target_quat
        else:
            target_pos, target_quat = self._get_delta_ee_pose_target()
        self._update_rotation_target_error(target_quat)

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            ctrl_target_gripper_dof_pos=self.cfg.ctrl.gripper_dof_pos,
        )

    def _apply_delta_joint_pos_action(self):
        """Apply a normalized joint delta as a direct arm joint target."""
        joint_threshold = torch.tensor(
            self.cfg.ctrl.joint_pos_threshold,
            device=self.device,
            dtype=self.joint_pos.dtype,
        ).reshape(1, 7)
        target = self.joint_pos[:, :7] + self.actions[:, :7] * joint_threshold
        self.ctrl_target_joint_pos[:, :7] = target
        self.ctrl_target_joint_pos[:, 7:9] = self.cfg.ctrl.gripper_dof_pos
        self.joint_torque[:, :] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _apply_abs_ee_pose_action(self, target_pose: torch.Tensor, clip_pose_target: bool = True):
        """Drive the controller toward an absolute fingertip pose target.

        `target_pose` is (num_envs, 7) = [pos(3), quat(4) wxyz] in the same
        frame as `fingertip_midpoint_pos/quat`. When `clip_pose_target` is True
        (default) the position and orientation errors are clipped to
        `pos_threshold`/`rot_threshold` with
        the same Euler + wrap_yaw logic as the standard absolute-pose action
        path, so the per-step target stays bounded.
        """
        target_pos = target_pose[:, 0:3]
        target_quat = target_pose[:, 3:7]
        target_quat = target_quat / torch.linalg.norm(target_quat, dim=-1, keepdim=True).clamp_min(1e-8)
        self._update_rotation_target_error(target_quat)

        # Position error (used for action_penalty logging too).
        self.delta_pos = target_pos - self.fingertip_midpoint_pos

        # Orientation error via Euler (same representation as the standard path).
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(target_quat)
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)
        self.delta_yaw = desired_yaw - curr_yaw

        if clip_pose_target:
            pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
            ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

            desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

            clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
            desired_xyz[:, 2] = curr_yaw + clipped_yaw

            desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
            desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

            delta_roll = desired_roll - curr_roll
            clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
            desired_xyz[:, 0] = curr_roll + clipped_roll

            curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
            desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

            delta_pitch = desired_pitch - curr_pitch
            clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
            desired_xyz[:, 1] = curr_pitch + clipped_pitch

            ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
                roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
            )
        else:
            ctrl_target_fingertip_midpoint_pos = target_pos
            ctrl_target_fingertip_midpoint_quat = target_quat

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.cfg.ctrl.gripper_dof_pos,
        )

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """Use the shared factory controller with optional task-space inertia."""
        self.last_ctrl_target_fingertip_midpoint_pos = ctrl_target_fingertip_midpoint_pos.detach().clone()
        self.last_ctrl_target_fingertip_midpoint_quat = ctrl_target_fingertip_midpoint_quat.detach().clone()

        task_prop = getattr(self, "task_prop_gains_matrix", None)
        task_deriv = getattr(self, "task_deriv_gains_matrix", None)
        if task_prop is None:
            task_prop = self.task_prop_gains
            task_deriv = self.task_deriv_gains

        nullspace_joint_target = None
        if self.cfg.ctrl.use_task_space_inertia:
            nullspace_joint_target = torch.tensor(
                self.cfg.ctrl.reset_joints, device=self.device, dtype=self.joint_pos.dtype
            ).repeat(self.num_envs, 1)

        self.joint_torque, self.applied_wrench, self.ctrl_debug = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=task_prop,
            task_deriv_gains=task_deriv,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
            nullspace_joint_target=nullspace_joint_target,
            return_debug=True,
            apply_task_inertia=self.cfg.ctrl.use_task_space_inertia,
        )

        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def dir_align_reward(self, a: float = 700, b: float = 0, tol: float = 0):
        """Reward the held asset for aligning its up axis with the fixed asset."""
        held_quat = self.held_quat
        fixed_quat = self.fixed_quat
        global_up = torch.tensor([[0, 0, 1.0]], device=held_quat.device)
        global_up_expanded = global_up.expand(fixed_quat.shape[0], 3)
        fixed_up_vec = quat_apply(fixed_quat, global_up_expanded)
        held_up_vec = quat_apply(held_quat, global_up_expanded)
        cos_sim = torch.sum(held_up_vec * fixed_up_vec, dim=1).clamp(-1.0, 1.0)
        align_error = torch.clamp(1.0 - cos_sim - tol, min=0.0)
        return factory_utils.squashing_fn(align_error, a, b)

    def ee_upright_error(self):
        """Return zero when the fingertip is top-down relative to the fixture."""
        global_up = torch.tensor([[0, 0, 1.0]], device=self.device).expand(self.num_envs, 3)
        fixed_up_vec = quat_apply(self.fixed_quat, global_up)
        fingertip_up_vec = quat_apply(self.fingertip_midpoint_quat, global_up)
        # A top-down fingertip has its local +Z axis anti-aligned with fixture +Z.
        topdown_cos = torch.sum((-fingertip_up_vec) * fixed_up_vec, dim=1).clamp(-1.0, 1.0)
        return 1.0 - topdown_cos

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_threshold = self.rot_threshold.clamp_min(1.0e-6)
        rot_error = torch.norm(self.delta_rot / rot_threshold, p=2, dim=-1)
        # Contact penalty.
        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        if self.cfg_task.contact_penalty_cap > 0:
            contact_penalty = torch.clamp(contact_penalty, max=self.cfg_task.contact_penalty_cap)
        ee_speed_threshold = max(float(self.cfg_task.ee_speed_penalty_threshold), 1e-6)
        ee_speed = torch.norm(self.fingertip_midpoint_linvel, p=2, dim=-1)
        self.ep_max_ee_speed = torch.maximum(self.ep_max_ee_speed, ee_speed.detach())

        ee_speed_penalty = torch.nn.functional.relu(ee_speed - self.cfg_task.ee_speed_penalty_threshold)
        ee_speed_excess_ratio = torch.clamp(
            ee_speed_penalty / ee_speed_threshold,
            max=self.cfg_task.ee_speed_exp_penalty_cap,
        )
        ee_speed_exp_penalty = torch.exp(self.cfg_task.ee_speed_exp_penalty_k * ee_speed_excess_ratio) - 1.0
        ee_ang_speed = torch.norm(self.fingertip_midpoint_angvel, p=2, dim=-1)
        ee_ang_speed_penalty = torch.nn.functional.relu(
            ee_ang_speed - self.cfg_task.ee_ang_speed_penalty_threshold
        )
        # Add success prediction rewards.
        
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, 
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # Delay success prediction penalty until some successes have occurred.
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        dir_align_rew = self.dir_align_reward()
        ee_upright_penalty = self.ee_upright_error()
        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "ee_speed_penalty": ee_speed_penalty,
            "ee_speed_exp_penalty": ee_speed_exp_penalty,
            "ee_ang_speed_penalty": ee_ang_speed_penalty,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
            "dir_align_rew": dir_align_rew,
            "ee_upright_penalty": ee_upright_penalty,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "ee_speed_penalty": -self.cfg_task.ee_speed_penalty_scale,
            "ee_speed_exp_penalty": -self.cfg_task.ee_speed_exp_penalty_scale,
            "ee_ang_speed_penalty": -self.cfg_task.ee_ang_speed_penalty_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
            "dir_align_rew": 1,
            "ee_upright_penalty": -self.cfg_task.ee_upright_penalty_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)
        self.current_action_rep = str(getattr(self.cfg.ctrl, "action_rep", "rel_ee_pose"))
        self.packet_stiffness_override = None
        self.packet_deriv_override = None
        self.packet_clip_pose_target = True

        # Compute initial relative action for correct EMA computation.
        if self.current_action_rep == "rel_ee_pose":
            fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
            pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
            pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
            self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

            # Relative yaw to bolt.
            unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            unrot_quat = torch_utils.quat_from_euler_xyz(
                roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
            )

            fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
            fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
            fingertip_yaw_bolt = torch.where(
                fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
            )
            fingertip_yaw_bolt = torch.where(
                fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
            )

            yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
            self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
            self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)
        if self.cfg.use_dead_zone:
            self.dead_zone_thresholds = (
                    torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
                )
        else:
            self.dead_zone_thresholds = None

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)
        self.ep_max_ee_speed[env_ids] = 0.0
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        self._log_episode_rewards(rew_dict)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            ep_max_ee_speed = self.ep_max_ee_speed[reset_ids]
            self.extras["max_ee_velocity/mean"] = ep_max_ee_speed.mean()
            self.extras["max_ee_velocity/max"] = ep_max_ee_speed.max()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
