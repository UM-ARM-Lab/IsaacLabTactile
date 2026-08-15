# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: control module.

Imported by base, environment, and task classes. Not directly executed.
"""

import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
    nullspace_joint_target=None,
    return_debug=False,
    apply_task_inertia=False,
    use_singularity_robust_inverse=False,
):
    """Compute Franka DOF torque to move fingertips towards target pose.

    When ``apply_task_inertia`` is True, premultiply the task-space PD wrench by
    the operational-space inertia before mapping it to joint torques. The default
    False value preserves the legacy Jacobian-transpose PD behavior.
    """
    # References:
    # 1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # 2) Modern Robotics

    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion
    task_wrench_raw = task_wrench.clone()

    # Offset task_wrench motion by random amount to simulate unreliability at low forces.
    # Check if absolute value is less than specified amount. If so, 0 out, otherwise, subtract.
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Operational-space inertia and dynamically consistent inverse used by both
    # full OSC and the nullspace projection.
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    task_inertia = jacobian @ arm_mass_matrix_inv @ jacobian_T
    if use_singularity_robust_inverse:
        task_inertia_det = torch.linalg.det(task_inertia)
        inverse_candidate = torch.inverse(task_inertia)
        pinv_candidate = torch.linalg.pinv(task_inertia, rcond=1.0e-2)
        use_inverse = task_inertia_det.abs().ge(1.0e-2).view(-1, 1, 1)
        arm_mass_matrix_task = torch.where(
            use_inverse, inverse_candidate, pinv_candidate
        )
    else:
        arm_mass_matrix_task = torch.inverse(task_inertia)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    if not torch.isfinite(arm_mass_matrix_task).all():
        bad = ~torch.isfinite(arm_mass_matrix_task).reshape(num_envs, -1).all(dim=-1)
        det = torch.linalg.det(task_inertia)
        raise RuntimeError(
            f"OSC task-inertia inverse (J M^-1 J^T)^-1 is non-finite in {int(bad.sum())}/{num_envs} envs "
            f"(near-singular arm configuration). min|det(J M^-1 J^T)|={det.abs().min().item():.3e}."
        )

    if apply_task_inertia:
        task_wrench = (arm_mass_matrix_task @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    torque_task = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)
    dof_torque[:, 0:7] = torque_task

    # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
    # roboticsproceedings.org/rss07/p31.pdf

    # Nullspace posture target: a per-env (num_envs, 7) tensor if provided (e.g. each
    # env's own reset joint configuration), otherwise the single global config from cfg.
    if nullspace_joint_target is not None:
        default_dof_pos_tensor = nullspace_joint_target.to(device)
    else:
        default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    # nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    if not torch.isfinite(dof_torque).all():
        bad = ~torch.isfinite(dof_torque).all(dim=-1)
        raise RuntimeError(
            f"OSC produced non-finite joint torques in {int(bad.sum())}/{num_envs} envs before clamping."
        )

    # TODO: Verify it's okay to no longer do gripper control here.
    dof_torque_pre_clamp = dof_torque.clone()
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    if return_debug:
        return dof_torque, task_wrench, {
            "pose_error": delta_fingertip_pose,
            "ee_vel": torch.cat((fingertip_midpoint_linvel, fingertip_midpoint_angvel), dim=-1),
            "task_wrench_raw": task_wrench_raw,
            "task_wrench_cmd": task_wrench,
            "torque_task": torque_task,
            "torque_null": torque_null.squeeze(-1),
            "dof_torque_pre_clamp": dof_torque_pre_clamp,
            "dead_zone_thresholds": dead_zone_thresholds,
            "task_inertia_diag": torch.diagonal(arm_mass_matrix_task, dim1=-2, dim2=-1),
            "task_inertia_det": torch.linalg.det(task_inertia).unsqueeze(-1),
            "task_inertia_used_pinv": (
                torch.linalg.det(task_inertia).abs().lt(1.0e-2).unsqueeze(-1)
            ),
        }
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == "pinv":  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1  # 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
):
    """Interpret PD gains as task-space gains. Apply to task-space error.

    `task_prop_gains`/`task_deriv_gains` may be either per-axis gains of shape
    (num_envs, 6) (diagonal stiffness, applied element-wise) or full task-space
    stiffness/damping matrices of shape (num_envs, 6, 6) (e.g. anisotropic
    directional compliance), applied as matrix-vector products.
    """
    # Matrix-valued gains: task_wrench = K @ pose_error - D @ vel.
    if task_prop_gains.dim() == 3:
        vel = torch.cat((fingertip_midpoint_linvel, fingertip_midpoint_angvel), dim=-1)
        task_wrench = (task_prop_gains @ delta_fingertip_pose.unsqueeze(-1)).squeeze(-1) + (
            task_deriv_gains @ (0.0 - vel).unsqueeze(-1)
        ).squeeze(-1)
        return task_wrench

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rot error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench
