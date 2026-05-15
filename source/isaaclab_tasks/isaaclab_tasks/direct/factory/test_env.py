# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test environment with Franka robot, operational space control, and a fixed peg."""

import numpy as np
import torch
from pathlib import Path

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat
from isaaclab.sensors import VisuoTactileSensor
from . import factory_utils
from .factory_env import (
    FactoryEnv,
    _collect_meshes_from_stage,
    _load_meshes_from_usd,
    _sample_meshes_to_points,
)
from .factory_env_cfg import FactoryTaskTestCfg, ASSET_DIR, OBS_DIM_CFG, STATE_DIM_CFG


class TestEnv(FactoryEnv):
    """Simple test environment with Franka robot and operational space control.
    
    This environment provides end-effector control using operational space control (OSC),
    matching the control method used in other factory environments (PegInsert, GearMesh, NutThread).
    A Peg8mm asset is spawned kinematically fixed on the table (same USD as PegInsert).
    
    Action space: 7 DOF 
        - actions[0:3]: end-effector position displacement (delta from current)
        - actions[3:6]: end-effector rotation displacement as axis-angle (delta from current)
        - actions[6]: gripper position target (normalized [-1, 1] maps to [0.0, 0.04] where -1=closed, 1=open)
    
    Control method: Operational Space Control (OSC)
        - Actions are interpreted as end-effector pose deltas
        - Computes task-space forces using PD control
        - Maps to joint torques via Jacobian transpose: τ = J^T * F
        - Applies torques directly (torque control mode, stiffness=0)
    """
    
    cfg: FactoryTaskTestCfg

    def __init__(self, cfg: FactoryTaskTestCfg, render_mode: str | None = None, **kwargs):
        # Update observation/state space based on obs_order and state_order
        base_obs_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.observation_space = base_obs_space + cfg.action_space  # Add prev_actions
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order]) + cfg.action_space
        
        # Call parent __init__ which will call _setup_scene, _init_tensors, etc.
        super().__init__(cfg, render_mode, **kwargs)
        self._sync_peg_pose_buffers()

    def _sync_peg_pose_buffers(self):
        """Update fixed/held peg pose buffers from the sim (or task init pose before first step)."""
        if self._fixed_peg is not None and hasattr(self._fixed_peg, "data"):
            self.fixed_pos = self._fixed_peg.data.root_pos_w - self.scene.env_origins
            self.fixed_quat = self._fixed_peg.data.root_quat_w
        else:
            init_pos = torch.tensor(self.cfg_task.fixed_peg_init_pos, device=self.device, dtype=torch.float32)
            init_rot = torch.tensor(self.cfg_task.fixed_peg_init_rot, device=self.device, dtype=torch.float32)
            self.fixed_pos = init_pos.unsqueeze(0).repeat(self.num_envs, 1)
            self.fixed_quat = init_rot.unsqueeze(0).repeat(self.num_envs, 1)
        self.held_pos = self.fixed_pos
        self.held_quat = self.fixed_quat

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions (without assets)."""
        # Set task gains (reuse parent method's logic)
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = self.default_gains

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)

        if self._fixed_peg is not None:
            peg_friction = self.cfg_task.gripper_peg_friction
            if peg_friction is None:
                peg_friction = self.cfg_task.held_asset_cfg.friction
            factory_utils.set_friction(self._fixed_peg, peg_friction, self.scene.num_envs)

    def _setup_scene(self):
        """Initialize simulation scene with robot and a kinematically fixed peg."""
        # Spawn ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Spawn table
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )
        
        # Spawn robot
        robot_usd_file = "franka_gelsight_r15_assembled.usd" if self.cfg.use_gelsight_finger else "franka_mimic.usd"
        self.cfg.robot.spawn.usd_path = f"{ASSET_DIR}/{robot_usd_file}"
        self._robot = Articulation(self.cfg.robot)

        # Same peg USD as PegInsert, but kinematic so it stays fixed on the table.
        self._fixed_peg = RigidObject(self.cfg_task.fixed_peg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["fixed_peg"] = self._fixed_peg

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add tactile sensor(s) if enabled
        if self.cfg.enable_tactile_sensor:
            print("[INFO] Enabling tactile sensor")
            self._tactile_cam = VisuoTactileSensor(self.cfg.tactile_cam)
            self.scene.sensors["tactile_cam"] = self._tactile_cam
            if self.cfg.enable_tactile_sensor_right:
                self._tactile_cam_right = VisuoTactileSensor(self.cfg.tactile_cam_right)
                self.scene.sensors["tactile_cam_right"] = self._tactile_cam_right
                print("[INFO] Right finger tactile sensor enabled")
            else:
                self._tactile_cam_right = None
                print("[INFO] Right finger tactile sensor disabled")
        else:
            print("[INFO] Disabling tactile sensor")
            self._tactile_cam = None
            self._tactile_cam_right = None
            
        if self.cfg.use_compliant_gripper and self._tactile_cam is not None:
            VisuoTactileSensor.setup_compliant_materials(self.cfg.tactile_cam)
            if self.cfg.enable_tactile_sensor_right and hasattr(self.cfg, "tactile_cam_right"):
                VisuoTactileSensor.setup_compliant_materials(self.cfg.tactile_cam_right)

        print(f"[INFO] Test environment created with {self.scene.num_envs} environments")

    def _reset_fixed_peg(self, env_ids: torch.Tensor):
        """Write the fixed peg back to its nominal pose (kinematic, env-frame init)."""
        init_pos = torch.tensor(self.cfg_task.fixed_peg_init_pos, device=self.device, dtype=torch.float32)
        init_rot = torch.tensor(self.cfg_task.fixed_peg_init_rot, device=self.device, dtype=torch.float32)
        peg_state = self._fixed_peg.data.default_root_state.clone()[env_ids]
        peg_state[:, 0:3] = init_pos.unsqueeze(0) + self.scene.env_origins[env_ids]
        peg_state[:, 3:7] = init_rot.unsqueeze(0)
        peg_state[:, 7:] = 0.0
        self._fixed_peg.write_root_pose_to_sim(peg_state[:, 0:7], env_ids=env_ids)
        self._fixed_peg.write_root_velocity_to_sim(peg_state[:, 7:], env_ids=env_ids)
        self._fixed_peg.reset()

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors."""
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins

        self._sync_peg_pose_buffers()
        self.fingertip_midpoint_pos_fixed = self.fingertip_midpoint_pos - self.fixed_pos
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing for velocities (more reliable)
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        if self.cfg.include_contact_forces:
            left_data = self._left_finger_contact_sensor.data
            self.left_finger_force[:, :] = left_data.net_forces_w[:, 0, :]
            right_data = self._right_finger_contact_sensor.data
            self.right_finger_force[:, :] = right_data.net_forces_w[:, 0, :]

            link_wrenches_parent = self._robot.root_physx_view.get_link_incoming_joint_force()
            parent_quat_w = self._robot.data.body_quat_w
            left_wrench_p = link_wrenches_parent[:, self.left_finger_body_idx]
            right_wrench_p = link_wrenches_parent[:, self.right_finger_body_idx]
            left_parent_quat = parent_quat_w[:, self.left_finger_parent_body_idx]
            right_parent_quat = parent_quat_w[:, self.right_finger_parent_body_idx]
            self.left_finger_wrench[:, 0:3] = torch_utils.quat_apply(left_parent_quat, left_wrench_p[:, 0:3])
            self.left_finger_wrench[:, 3:6] = torch_utils.quat_apply(left_parent_quat, left_wrench_p[:, 3:6])
            self.right_finger_wrench[:, 0:3] = torch_utils.quat_apply(right_parent_quat, right_wrench_p[:, 0:3])
            self.right_finger_wrench[:, 3:6] = torch_utils.quat_apply(right_parent_quat, right_wrench_p[:, 3:6])

        if self.cfg.include_tactile_pointclouds:
            if self._pc_left_meshes is None or self._pc_right_meshes is None or self._pc_peg_meshes is None:
                robot_usd = Path(ASSET_DIR) / "franka_gelsight_r15_assembled.usd"
                peg_usd = Path(ASSET_DIR) / "factory_peg_8mm.usd"
                self._pc_left_meshes = _load_meshes_from_usd(robot_usd, "/panda/panda_leftfinger/elastomer")
                self._pc_right_meshes = _load_meshes_from_usd(robot_usd, "/panda/panda_rightfinger/elastomer")
                if not self._pc_left_meshes or not self._pc_right_meshes:
                    meshes = _load_meshes_from_usd(robot_usd, "/panda")
                    self._pc_left_meshes = meshes
                    self._pc_right_meshes = meshes

                if peg_usd.exists():
                    from pxr import Usd

                    stage = Usd.Stage.Open(str(peg_usd))  # pyright: ignore[reportAttributeAccessIssue]
                    default_prim = stage.GetDefaultPrim()
                    root = default_prim.GetPath().pathString if default_prim else "/"
                    self._pc_peg_meshes = _collect_meshes_from_stage(stage, root)
                else:
                    self._pc_peg_meshes = []

            half = max(1, self.cfg.tactile_pointcloud_gripper_points // 2)
            left_pts_np = _sample_meshes_to_points(self._pc_left_meshes, half)
            right_pts_np = _sample_meshes_to_points(
                self._pc_right_meshes, self.cfg.tactile_pointcloud_gripper_points - half
            )
            peg_pts_np = _sample_meshes_to_points(self._pc_peg_meshes, self.cfg.tactile_pointcloud_peg_points)

            if left_pts_np.size > 0 and right_pts_np.size > 0 and peg_pts_np.size > 0:
                left_pts_l = torch.tensor(left_pts_np, dtype=torch.float32, device=self.device)
                right_pts_l = torch.tensor(right_pts_np, dtype=torch.float32, device=self.device)
                peg_pts_l = torch.tensor(peg_pts_np, dtype=torch.float32, device=self.device)

                left_pos_e = self._robot.data.body_pos_w[:, self.left_finger_body_idx] - self.scene.env_origins
                left_quat_w = self._robot.data.body_quat_w[:, self.left_finger_body_idx]
                right_pos_e = self._robot.data.body_pos_w[:, self.right_finger_body_idx] - self.scene.env_origins
                right_quat_w = self._robot.data.body_quat_w[:, self.right_finger_body_idx]
                peg_pos_e = self.fixed_pos
                peg_quat_w = self.fixed_quat

                def _apply_pc(quat_w, pts_l, pos_e):
                    e_count, p_count = quat_w.shape[0], pts_l.shape[0]
                    quat_exp = quat_w.unsqueeze(1).expand(e_count, p_count, 4).reshape(-1, 4)
                    pts_exp = pts_l.unsqueeze(0).expand(e_count, p_count, 3).reshape(-1, 3)
                    pos_exp = pos_e.unsqueeze(1).expand(e_count, p_count, 3).reshape(-1, 3)
                    pc = torch_utils.quat_apply(quat_exp, pts_exp) + pos_exp
                    return pc.view(e_count, p_count, 3)

                self.tactile_pc_left_w = _apply_pc(left_quat_w, left_pts_l, left_pos_e)
                self.tactile_pc_right_w = _apply_pc(right_quat_w, right_pts_l, right_pos_e)
                self.tactile_pc_peg_w = _apply_pc(peg_quat_w, peg_pts_l, peg_pos_e)

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic (without asset references)."""
        prev_actions = self.actions.clone()
        fingertip_rot_mat = matrix_from_quat(self.fingertip_midpoint_quat)
        # 6D orientation representation using the first two rotation matrix columns.
        fingertip_orn_6d = torch.cat((fingertip_rot_mat[:, :, 0], fingertip_rot_mat[:, :, 1]), dim=-1)

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos_fixed,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_orn_6d": fingertip_orn_6d,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "gripper_pos": self.joint_pos[:, 7:9],
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos_fixed,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_orn_6d": fingertip_orn_6d,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _apply_action(self):
        """Apply actions using operational space control (OSC) - simplified version without asset clipping."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as absolute target end-effector position in env frame.
        ctrl_target_fingertip_midpoint_pos = self.actions[:, 0:3]

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        # Convert rotation actions to quaternion
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-8)  # Avoid division by zero

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # Interpret gripper action: map from [-1, 1] to [0.0, 0.04] where -1=closed, 1=open
        gripper_actions = self.actions[:, 6]  # Normalized action in [-1, 1]
        ctrl_target_gripper_dof_pos = (gripper_actions + 1.0) * 0.02  # Maps [-1, 1] to [0.0, 0.04]
        ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, 0.0, 0.04)  # Ensure valid range
        # Expand to [num_envs, 2] to match the two gripper DOFs (indices 7:9)
        ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos.unsqueeze(-1).expand(-1, 2)

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
        )

    def _get_rewards(self) -> torch.Tensor:
        """Simple reward for test environment - just return zeros."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset robot to initial position (without assets).
        
        If hand_init_pos is specified (not all zeros), uses IK to solve for joint angles
        to achieve the desired end-effector pose. Otherwise, uses default joint positions.
        """
        # Call DirectRLEnv._reset_idx directly (skipping FactoryEnv._reset_idx which handles assets)
        from isaaclab.envs import DirectRLEnv
        DirectRLEnv._reset_idx(self, env_ids)
        
        # Check if we should use IK-based initialization
        hand_init_pos = torch.tensor(self.cfg_task.hand_init_pos, device=self.device)
        use_ik_init = torch.any(torch.abs(hand_init_pos) > 1e-6)  # Check if not all zeros
        
        if use_ik_init:
            # Use IK to solve for joint angles given desired end-effector pose
            # First set robot to default pose (use 0.04 for gripper since test env has no held asset)
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_pos[:, :7] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)[None, :]
            joint_pos[:, 7:] = 0.04  # Open gripper
            joint_vel = torch.zeros_like(joint_pos)
            self.ctrl_target_joint_pos[env_ids, :] = joint_pos
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self._robot.reset()
            self.step_sim_no_action()
            
            # Compute target end-effector position in environment-local coordinates
            # hand_init_pos is relative to robot base (which is at scene.env_origins)
            # Note: IK solver expects environment-local coordinates (matching fingertip_midpoint_pos)
            target_ee_pos = hand_init_pos.unsqueeze(0).repeat(len(env_ids), 1)
            
            # Apply optional position noise
            if torch.any(torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device) > 1e-6):
                rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                pos_noise = pos_noise @ torch.diag(hand_init_pos_rand)
                target_ee_pos += pos_noise
            
            # Compute target end-effector orientation
            hand_init_orn_euler = torch.tensor(
                self.cfg_task.hand_init_orn, device=self.device
            ).unsqueeze(0).repeat(len(env_ids), 1)
            
            # Apply optional orientation noise
            if torch.any(torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device) > 1e-6):
                rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
                orn_noise = orn_noise @ torch.diag(hand_init_orn_rand)
                hand_init_orn_euler += orn_noise
            
            target_ee_quat = torch_utils.quat_from_euler_xyz(
                roll=hand_init_orn_euler[:, 0],
                pitch=hand_init_orn_euler[:, 1],
                yaw=hand_init_orn_euler[:, 2]
            )
            
            # Solve IK iteratively
            bad_envs = env_ids.clone()
            ik_attempt = 0
            max_ik_attempts = 10
            
            while bad_envs.shape[0] > 0 and ik_attempt < max_ik_attempts:
                # Solve IK for problematic environments
                pos_error, aa_error = self.set_pos_inverse_kinematics(
                    ctrl_target_fingertip_midpoint_pos=target_ee_pos,
                    ctrl_target_fingertip_midpoint_quat=target_ee_quat,
                    env_ids=bad_envs,
                )
                
                # Check if IK succeeded
                pos_error_norm = torch.linalg.norm(pos_error, dim=1)
                angle_error_norm = torch.norm(aa_error, dim=1)
                pos_error_check = pos_error_norm > 1e-3
                angle_error_check = angle_error_norm > 1e-3
                any_error = torch.logical_or(pos_error_check, angle_error_check)
                
                # Update bad_envs for next iteration
                bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]
                
                if bad_envs.shape[0] == 0:
                    break
                
                # Reset failed environments to default pose and try again
                joint_pos = self._robot.data.default_joint_pos[bad_envs]
                joint_pos[:, :7] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)[None, :]
                joint_pos[:, 7:] = 0.04  # Open gripper
                joint_vel = torch.zeros_like(joint_pos)
                self.ctrl_target_joint_pos[bad_envs, :] = joint_pos
                self._robot.set_joint_position_target(self.ctrl_target_joint_pos[bad_envs], env_ids=bad_envs)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=bad_envs)
                self._robot.reset()
                self.step_sim_no_action()
                ik_attempt += 1
            
            # Set gripper to open position after IK (IK only sets arm joints 0:7)
            self.joint_pos[env_ids, 7:9] = 0.04  # Open gripper
            self.ctrl_target_joint_pos[env_ids, 7:9] = 0.04
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
            
            if bad_envs.shape[0] > 0:
                print(f"[WARNING] IK failed for {bad_envs.shape[0]} environments after {ik_attempt} attempts. Using default joint positions.")
                # Fall back to default joints for failed environments
                joint_pos = self._robot.data.default_joint_pos[bad_envs]
                joint_pos[:, :7] = torch.tensor(
                    self.cfg.ctrl.reset_joints, device=self.device
                )[None, :]
                joint_pos[:, 7:] = 0.04  # Open gripper
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=bad_envs)
                self._robot.reset()
                self.ctrl_target_joint_pos[bad_envs] = joint_pos
        else:
            # Use default joint positions (original behavior)
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_pos[:, :7] = torch.tensor(
                self.cfg.ctrl.reset_joints, device=self.device
            )[None, :]
            joint_pos[:, 7:] = 0.04  # Open gripper
            
            joint_vel = torch.zeros_like(joint_pos)
            
            # Write to simulation
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self._robot.reset()
            
            # Update control targets
            self.ctrl_target_joint_pos[env_ids] = joint_pos
        
        # Reset finite-differencing buffers
        self.prev_joint_pos[env_ids] = self.joint_pos[env_ids, 0:7]
        
        # Compute intermediate values to get fingertip pose for reset buffers
        self.step_sim_no_action()
        
        # Now update fingertip buffers
        self.prev_fingertip_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self.prev_fingertip_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
        
        # Reset actions
        self.actions[env_ids] = 0.0
        if hasattr(self, 'prev_actions'):
            self.prev_actions[env_ids] = 0.0
        
        # Zero initial velocity
        self.ee_angvel_fd[env_ids] = 0.0
        self.ee_linvel_fd[env_ids] = 0.0

        self._reset_fixed_peg(env_ids)

        # Get initial tactile render if enabled
        if self.cfg.enable_tactile_sensor and self._tactile_cam is not None:
            if self._tactile_cam._nominal_tactile is None:
                self.sim.render()
                self._tactile_cam.get_initial_render()
            if self.cfg.enable_tactile_sensor_right and self._tactile_cam_right is not None:
                if self._tactile_cam_right._nominal_tactile is None:
                    self._tactile_cam_right.get_initial_render()
