#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
# noqa: SLF001

"""
Demo script for Factory Peg Insertion that sets the environment to a provided state.

This script:
 - Instantiates the Isaac-Factory-PegInsert-Direct-v0 task with a single environment
 - Sets the environment to a provided state (defaults to zeros, user can provide custom state)
 - Steps the environment with zero actions
 - Visualizes the scene via the Isaac Sim viewer

Usage:
    python scripts/demos/factory/peg_insert_set_state.py \
        --enable_cameras --print_tactile
    
    # With custom state (provide state file or use zeros):
    python scripts/demos/factory/peg_insert_set_state.py \
        --enable_cameras --state_file /path/to/state.npz

Notes:
 - AppLauncher must be called first to set up Omniverse environment
 - Use --enable_cameras to render the scene
 - State format: dictionary with keys "articulation", "rigid_object", etc.
   Each entity has state components like root_pose, root_velocity, joint_position, joint_velocity
 - If no state is provided, uses zeros for all state components
"""

import argparse
import os
import numpy as np
import torch
from typing import cast

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Factory Peg Insertion with state setting")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 = run forever)")
parser.add_argument("--print_tactile", action="store_true", help="Print basic tactile stats each step")
parser.add_argument("--compliance_stiffness", type=float, default=350.0, help="Compliance stiffness for tactile sensor")
parser.add_argument("--state_file", type=str, default=None, help="Path to state file (.npz) to load. If not provided, uses zeros.")
parser.add_argument("--use_zeros", action="store_true", default=False, help="Use zeros for state if state_file not provided (default: True)")
# parser.add_argument("--save_state", type=str, default=None, help="Path to save current state to (.npz file). Useful for understanding state format.")
parser.add_argument("--save_state", type=str, default='./temp.npz', help="Path to save current state to (.npz file). Useful for understanding state format.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# Launch omniverse app (MUST be before importing other Isaac Lab modules)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of script follows after AppLauncher setup."""

import gymnasium as gym

# Register isaaclab_tasks environments (must import to register gym tasks)
import isaaclab_tasks as _isaaclab_tasks  # noqa: F401

from isaaclab.utils.timer import Timer
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
from omegaconf import OmegaConf


def create_zero_state(scene, device: str = "cuda:0") -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Create a state dictionary with zeros for all entities in the scene.
    
    For peg insertion task, the scene contains:
    - Articulations: "robot", "fixed_asset", "held_asset"
    - No deformable objects, rigid objects, or surface grippers
    
    Args:
        scene: The InteractiveScene object
        device: Device to create tensors on
        
    Returns:
        State dictionary with zero values for all entities
    """
    state = {}
    num_envs = scene.num_envs
    
    # Articulations (robot, fixed_asset, held_asset, and optionally gears for gear_mesh task)
    state["articulation"] = {}
    for asset_name, articulation in scene.articulations.items():
        num_joints = articulation.num_joints
        asset_state = {
            "root_pose": torch.zeros((num_envs, 7), device=device),  # [x, y, z, qw, qx, qy, qz]
            "root_velocity": torch.zeros((num_envs, 6), device=device),  # [lin_vel, ang_vel]
            "joint_position": torch.zeros((num_envs, num_joints), device=device),
            "joint_velocity": torch.zeros((num_envs, num_joints), device=device),
        }
        # Set quaternion to identity [1, 0, 0, 0] for root_pose
        asset_state["root_pose"][:, 3] = 1.0  # qw = 1.0
        state["articulation"][asset_name] = asset_state
    
    # Rigid objects (not present in peg insertion, but included for generality)
    state["rigid_object"] = {}
    for asset_name, _rigid_object in scene.rigid_objects.items():
        asset_state = {
            "root_pose": torch.zeros((num_envs, 7), device=device),
            "root_velocity": torch.zeros((num_envs, 6), device=device),
        }
        # Set quaternion to identity
        asset_state["root_pose"][:, 3] = 1.0
        state["rigid_object"][asset_name] = asset_state
    
    # Deformable objects (not present in peg insertion, but included for generality)
    state["deformable_object"] = {}
    for asset_name, deformable_object in scene.deformable_objects.items():
        # Get number of nodes from the deformable object
        num_nodes = deformable_object.data.nodal_pos_w.shape[1]
        asset_state = {
            "nodal_position": torch.zeros((num_envs, num_nodes, 3), device=device),
            "nodal_velocity": torch.zeros((num_envs, num_nodes, 3), device=device),
        }
        state["deformable_object"][asset_name] = asset_state
    
    # Surface grippers (not present in peg insertion, but included for generality)
    state["gripper"] = {}
    for asset_name, gripper in scene.surface_grippers.items():
        # Get gripper state shape
        gripper_state_shape = gripper.state.shape
        state["gripper"][asset_name] = torch.zeros(gripper_state_shape, device=device)
    
    return state


def save_state_to_file(state: dict[str, dict[str, dict[str, torch.Tensor]]], state_file: str):
    """Save state dictionary to a .npz file.
    
    Args:
        state: State dictionary with torch tensors
        state_file: Path to save .npz file
    """
    npz_data = {}
    
    # Convert torch tensors to numpy and save
    for category, entities in state.items():
        category_data = {}
        for entity_name, entity_state in entities.items():
            entity_data = {}
            for key, value in entity_state.items():
                if isinstance(value, torch.Tensor):
                    entity_data[key] = value.detach().cpu().numpy()
                else:
                    entity_data[key] = value
            category_data[entity_name] = entity_data
        npz_data[category] = category_data
    
    np.savez(state_file, **npz_data)
    print(f"[INFO] State saved to: {state_file}")


def load_state_from_file(state_file: str, device: str = "cuda:0") -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Load state from a .npz file and convert to torch tensors.
    
    Args:
        state_file: Path to .npz file containing state
        device: Device to create tensors on
        
    Returns:
        State dictionary with torch tensors
    """
    # Load numpy arrays from file
    npz_data = np.load(state_file, allow_pickle=True)
    
    # Convert to state dictionary format
    state = {}
    
    # Load articulations
    if "articulation" in npz_data:
        state["articulation"] = {}
        articulation_data = npz_data["articulation"].item() if isinstance(npz_data["articulation"].item(), dict) else npz_data["articulation"]
        for asset_name, asset_data in articulation_data.items():
            asset_state = {}
            for key in ["root_pose", "root_velocity", "joint_position", "joint_velocity"]:
                if key in asset_data:
                    asset_state[key] = torch.from_numpy(np.array(asset_data[key])).to(device)
            state["articulation"][asset_name] = asset_state
    
    # Load rigid objects
    if "rigid_object" in npz_data:
        state["rigid_object"] = {}
        rigid_data = npz_data["rigid_object"].item() if isinstance(npz_data["rigid_object"].item(), dict) else npz_data["rigid_object"]
        for asset_name, asset_data in rigid_data.items():
            asset_state = {}
            for key in ["root_pose", "root_velocity"]:
                if key in asset_data:
                    asset_state[key] = torch.from_numpy(np.array(asset_data[key])).to(device)
            state["rigid_object"][asset_name] = asset_state
    
    # Load deformable objects
    if "deformable_object" in npz_data:
        state["deformable_object"] = {}
        deformable_data = npz_data["deformable_object"].item() if isinstance(npz_data["deformable_object"].item(), dict) else npz_data["deformable_object"]
        for asset_name, asset_data in deformable_data.items():
            asset_state = {}
            for key in ["nodal_position", "nodal_velocity"]:
                if key in asset_data:
                    asset_state[key] = torch.from_numpy(np.array(asset_data[key])).to(device)
            state["deformable_object"][asset_name] = asset_state
    
    # Load grippers
    if "gripper" in npz_data:
        state["gripper"] = {}
        gripper_data = npz_data["gripper"].item() if isinstance(npz_data["gripper"].item(), dict) else npz_data["gripper"]
        for asset_name, asset_data in gripper_data.items():
            state["gripper"][asset_name] = torch.from_numpy(np.array(asset_data)).to(device)
    
    return state


def main():
    # Create a modified config that enables tactile sensing
    env_cfg = FactoryTaskPegInsertCfg(params=OmegaConf.create({"env": {}}))
    env_cfg.enable_tactile_sensor = True
    env_cfg.read_tactile_sensor = True
    env_cfg.enable_obs_camera = False
    env_cfg.use_compliant_gripper = True
    env_cfg.use_gelsight_finger = True
    env_cfg.scene.num_envs = 1  # Only one environment
    env_cfg.tactile_cam.compliance_stiffness = args_cli.compliance_stiffness

    # Update simulation device
    if args_cli.device == "cpu":
        env_cfg.sim.device = "cpu"
    else:
        env_cfg.sim.device = args_cli.device

    # Create environment via gymnasium registry
    print("[INFO] Creating Factory Peg Insert environment with tactile sensor...")
    env: FactoryEnv = gym.make(
        "Isaac-Factory-PegInsert-Direct-v0",
        cfg=env_cfg,
    )
    print("[INFO] Environment created successfully.")

    # Reset environment once to initialize
    _, _ = env.reset()
    print("[INFO] Environment reset.")

    # Get the scene to access state structure
    scene = env.unwrapped.scene
    device = env.unwrapped.device
    
    # Save current state if requested (before modifying it)
    if args_cli.save_state:
        current_state = scene.get_state(is_relative=False)
        save_state_to_file(current_state, args_cli.save_state)
        print("[INFO] Current state saved. You can modify this file and load it with --state_file")
    
    # Get or create state
    if args_cli.state_file and os.path.exists(args_cli.state_file):
        print(f"[INFO] Loading state from file: {args_cli.state_file}")
        state = load_state_from_file(args_cli.state_file, device=device)
    else:
        if args_cli.use_zeros:
            print("[INFO] Creating zero state (all values set to 0, quaternions set to identity)...")
            state = create_zero_state(scene, device=device)
        else:
            # Get current state as template
            print("[INFO] Using current environment state as template...")
            state = scene.get_state(is_relative=False)
    
    # Set environment to the provided state
    print("[INFO] Setting environment to provided state...")
    env_ids = torch.tensor([0], dtype=torch.int64, device=device)  # Only environment 0
    
    # Reset the environment first (this calls _reset_idx internally and resets internal buffers)
    # Note: We need to reset to initialize all internal state before setting custom state
    # Using protected method is necessary here as DirectRLEnv doesn't expose reset_to
    env.unwrapped._reset_idx(env_ids)  # noqa: SLF001
    
    # Now set the scene to the desired state
    # This will override the positions/velocities set by the reset
    # Note: scene.reset_to() internally calls write_data_to_sim()
    scene.reset_to(state, env_ids=env_ids, is_relative=False)
    
    # Step simulation forward to apply the state changes
    # This ensures physics constraints are resolved and the state is stable
    env.unwrapped.sim.forward()
    
    # Update scene buffers to reflect any adjustments made by the simulation
    # (e.g., collision resolution, constraint satisfaction)
    # This ensures buffers match the actual simulation state after forward()
    scene.update(dt=env.unwrapped.physics_dt)
    
    # Manually update the camera view to reflect the new asset position after state reset
    # This is necessary because the camera callback may not update immediately or may use stale data
    if env.unwrapped.viewport_camera_controller is not None:
        vcc = env.unwrapped.viewport_camera_controller
        # Check if camera is tracking an asset and update accordingly
        if vcc.cfg.origin_type == "asset_root" and vcc.cfg.asset_name is not None:
            vcc.update_view_to_asset_root(vcc.cfg.asset_name)
        elif vcc.cfg.origin_type == "asset_body" and vcc.cfg.asset_name is not None and vcc.cfg.body_name is not None:
            vcc.update_view_to_asset_body(vcc.cfg.asset_name, vcc.cfg.body_name)
        elif vcc.cfg.origin_type == "env":
            vcc.update_view_to_env()
    
    # Render if cameras are enabled
    if env.unwrapped.sim.has_rtx_sensors() and env_cfg.rerender_on_reset:
        env.unwrapped.sim.render()
    
    print("[INFO] Environment state set successfully.")

    # Zero action loop
    action_dim = env.unwrapped.action_size if hasattr(env.unwrapped, "action_size") else 6
    random_actions = torch.randn((1, action_dim), dtype=torch.float32, device=device)
    print("[INFO] Starting viewer loop. Press Ctrl+C or close viewer to exit.")

    timer = Timer()
    timer.start()
    steps = 0

    try:
        while simulation_app.is_running():
            # Step environment with zero actions
            _obs_next, _rewards, _terminateds, _truncateds, _extras = env.step(random_actions)
            
            # Note: Camera automatically tracks asset via callback at each render step
            # The callback reads scene[asset_name].data.root_pos_w[env_index] and updates camera view
            # So the camera should continue to follow the asset even after state changes
            
            # Print tactile stats if requested
            if args_cli.print_tactile:
                unwrapped_env: FactoryEnv = env.unwrapped  # type: ignore[assignment]
                tactile_cam = cast(any, getattr(unwrapped_env, "_tactile_cam", None))  # noqa: SLF001
                if hasattr(unwrapped_env, "_tactile_cam") and tactile_cam is not None:
                    data = tactile_cam.data  # noqa: SLF001
                    if hasattr(data, "taxim_tactile") and data.taxim_tactile is not None:
                        taxim = data.taxim_tactile
                        taxim_cpu = taxim.detach().cpu().numpy()
                        print(f"[TACTILE] Shape: {taxim_cpu.shape}, Min: {taxim_cpu.min():.3f}, Max: {taxim_cpu.max():.3f}, Mean: {taxim_cpu.mean():.3f}")

            # FPS logging
            steps += 1
            if args_cli.steps > 0 and steps >= args_cli.steps:
                break
    finally:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

