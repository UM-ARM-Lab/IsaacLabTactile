# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple test environment with Franka robot and operational space control."""

import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.sensors import VisuoTactileSensor
from . import factory_utils
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskTestCfg, ASSET_DIR, OBS_DIM_CFG, STATE_DIM_CFG


class TestEnv(FactoryEnv):
    """Simple test environment with Franka robot and operational space control.
    
    This environment provides end-effector control using operational space control (OSC),
    matching the control method used in other factory environments (PegInsert, GearMesh, NutThread).
    No objects or manipulation tasks are included - just the robot on a table.
    
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
        
        # Note: Skip friction setting for assets since test environment has no assets

    def _setup_scene(self):
        """Initialize simulation scene (without fixed/held assets)."""
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

        # Spawn fixed cylinder (peg-like object) below the gripper, fixed to the table
        # Cylinder size: radius=0.004m (8mm diameter), height=0.04m (4cm) - suitable for grasping
        # Position: (0.55, 0.0, 0.05) - centered on table, on table surface
        cylinder_cfg = sim_utils.CylinderCfg(
            radius=0.01,  # 8mm diameter peg
            height=0.1,   # 4cm tall
            axis="Z",      # Vertical cylinder
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Fixed to table, won't move
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # Small mass for collision
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.2)),  # Yellow color
        )
        cylinder_cfg.func(
            "/World/envs/env_.*/Cylinder",
            cylinder_cfg,
            translation=(0.49, 0.0, 0.05),  # On table surface, centered below gripper
            orientation=(1.0, 0.0, 0.0, 0.0),  # Upright
        )

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add tactile sensor if enabled
        if self.cfg.enable_tactile_sensor:
            print("[INFO] Enabling tactile sensor")
            self._tactile_cam = VisuoTactileSensor(self.cfg.tactile_cam)
            self.scene.sensors["tactile_cam"] = self._tactile_cam
        else:
            print("[INFO] Disabling tactile sensor")
            self._tactile_cam = None
            
        if self.cfg.use_compliant_gripper and self._tactile_cam is not None:
            VisuoTactileSensor.setup_compliant_materials(self.cfg.tactile_cam)

        print(f"[INFO] Test environment created with {self.scene.num_envs} environments")
        
        # Note: No fixed_asset or held_asset spawned for test environment

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors (without asset references)."""
        # Only compute robot-related intermediate values
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
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

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic (without asset references)."""
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "gripper_pos": self.joint_pos[:, 7:9],
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
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

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        # Compute target end-effector pose (no position clipping since no fixed asset)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

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

        # Restrict to upright orientation (same as other factory tasks)
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

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
        
        # Get initial tactile render if enabled
        if self.cfg.enable_tactile_sensor and self._tactile_cam is not None:
            if self._tactile_cam._nominal_tactile is None:
                self.sim.render()
                self._tactile_cam.get_initial_render()
