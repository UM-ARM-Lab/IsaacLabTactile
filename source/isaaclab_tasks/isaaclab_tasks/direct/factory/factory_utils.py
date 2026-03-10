# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils


def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Set robot gains using critical damping."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def wrap_yaw(angle):
    """Ensure yaw stays within range."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def set_friction(asset, value, num_envs):
    """Update material properties for a given asset. value is a scalar applied to all envs."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)


def set_friction_for_envs(asset, values, env_ids):
    """Set friction per env for the given env_ids. values: tensor of shape (len(env_ids),)."""
    env_ids = env_ids.reshape(-1)  # ensure 1D (avoid 0-dim when single env)
    if env_ids.numel() == 0:
        return
    materials = asset.root_physx_view.get_material_properties()
    values = values.to(device=materials.device, dtype=materials.dtype)
    env_ids_idx = env_ids.to(device=materials.device)
    # Support both 2D (num_envs, 3) and 3D (num_envs, num_shapes, 3) material buffers
    if materials.dim() == 2:
        materials[env_ids_idx, 0] = values
        materials[env_ids_idx, 1] = values
    else:
        materials[env_ids_idx, ..., 0] = values.unsqueeze(-1)
        materials[env_ids_idx, ..., 1] = values.unsqueeze(-1)
    # PhysX API expects env indices on CPU, contiguous long (consistent with set_friction)
    env_ids_cpu = env_ids.cpu().long().contiguous()
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def set_body_inertias(robot, num_envs):
    """Note: this is to account for the asset_options.armature parameter in IGE."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(num_envs))


def get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device):
    """Get transform between asset default frame and geometric base frame."""
    held_base_x_offset = 0.0
    if task_name == "peg_insert":
        held_base_z_offset = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif task_name == "nut_thread":
        held_base_z_offset = fixed_asset_cfg.base_height
    else:
        raise NotImplementedError("Task not implemented")

    held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset

    return held_base_pos_local


def get_held_base_pose(held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get current poses for keypoint and success computation."""
    held_base_pos_local = get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get target poses for keypoint and success computation."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if task_name == "peg_insert":
        fixed_success_pos_local[:, 2] = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        fixed_success_pos_local[:, 0] = gear_base_offset[0]
        fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif task_name == "nut_thread":
        head_height = fixed_asset_cfg.base_height
        shank_length = fixed_asset_cfg.height
        thread_pitch = fixed_asset_cfg.thread_pitch
        fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
    else:
        raise NotImplementedError("Task not implemented")
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x, a, b):
    """Compute bounded reward function."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict, obs_order):
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors


def change_FT_frame(source_F, source_T, source_frame, target_frame):
    """Convert force/torque reading from source to target frame."""
    # Modern Robotics eq. 3.95
    source_frame_inv = torch_utils.tf_inverse(source_frame[0], source_frame[1])
    target_T_source_quat, target_T_source_pos = torch_utils.tf_combine(
        source_frame_inv[0], source_frame_inv[1], target_frame[0], target_frame[1]
    )
    target_F = torch_utils.quat_apply(target_T_source_quat, source_F)
    target_T = torch_utils.quat_apply(
        target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1))
    )
    return target_F, target_T
