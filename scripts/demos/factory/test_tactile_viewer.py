#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""
Viewer demo for Factory Test environment with operational space control and tactile sensing.

This script:
 - Instantiates the Isaac-Factory-Test-Direct-v0 task
 - Enables the tactile sensor and reading of tactile images (optional)
 - Steps the environment with zero actions (robot stays in place)
 - Visualizes the scene via the Isaac Sim viewer
 - Optionally saves tactile images from each episode to video files

Usage:
    # Basic viewer with tactile sensor
    python scripts/demos/factory/test_tactile_viewer.py \\
        --num_envs 4 --enable_cameras
    
    # Save tactile videos for each episode:
    python scripts/demos/factory/test_tactile_viewer.py \\
        --num_envs 4 --enable_cameras --save_video ./test_videos --video_fps 20

Notes:
 - AppLauncher must be called first to set up Omniverse environment
 - Use --enable_cameras to render the scene
 - Tactile images are exposed via env.unwrapped._tactile_cam.data.taxim_tactile when enabled
 - Videos are saved per episode as tactile_episode_XXXX.mp4 in the specified directory
- Action space is 7 DOF: end-effector control (3D position + 3D rotation as axis-angle) + gripper control
- Control method: Operational Space Control (OSC) using torque-based control
"""

import argparse
import os
import numpy as np
import torch
from typing import cast
import cv2

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Factory Test environment viewer with tactile sensor")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
parser.add_argument("--steps", type=int, default=149, help="Number of steps to run (0 = run forever)")
parser.add_argument("--print_tactile", action="store_true", help="Print basic tactile stats each step")
parser.add_argument("--save_video", type=str, default=None, help="Directory to save tactile videos (one per episode)")
# parser.add_argument("--save_video", type=str, default="./test_videos", help="Directory to save tactile videos (one per episode)")
parser.add_argument("--video_fps", type=int, default=20, help="FPS for saved videos")

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
from isaaclab_tasks.direct.factory.test_env import TestEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskTestCfg
from omegaconf import OmegaConf


def main():
    # Create a modified config that enables tactile sensing
    env_cfg = FactoryTaskTestCfg(params=OmegaConf.create({"env": {}}))
    env_cfg.enable_tactile_sensor = True
    env_cfg.read_tactile_sensor = True
    env_cfg.enable_obs_camera = False
    env_cfg.use_compliant_gripper = True
    env_cfg.use_gelsight_finger = True
    env_cfg.scene.num_envs = args_cli.num_envs

    # Update simulation device
    if args_cli.device == "cpu":
        env_cfg.sim.device = "cpu"
    else:
        env_cfg.sim.device = args_cli.device

    # Create environment via gymnasium registry
    print("[INFO] Creating Factory Test environment with tactile sensor...")
    env: TestEnv = gym.make(
        "Isaac-Factory-Test-Direct-v0",
        cfg=env_cfg,
    )
    print("[INFO] Environment created successfully.")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # Setup video recording if enabled
    video_output_dir = None
    if args_cli.save_video:
        video_output_dir = args_cli.save_video
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"[INFO] Videos will be saved to: {video_output_dir}")

    # Reset once before stepping
    _, _ = env.reset()
    print("[INFO] Environment reset.")

    # Video recording state
    episode_num = 0
    episode_frames = []  # Store frames for current episode
    video_writer = None  # Current video writer
    video_height = None
    video_width = None

    # Zero action loop (7 DOF: end-effector control via operational space control + gripper)
    down_actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float32)
    # down_actions[:, 2] = -0.3  # Small downward position offset
    down_actions[:, 6] = 0.0  # Keep gripper open (1.0 maps to 0.04, fully open)

    close_actions = down_actions.clone()
    close_actions[:, 6] = -1.0  # Keep gripper closed (-1.0 maps to -0.04, fully closed)
    close_actions[:, 2] = 0.0

    down_actions[:, 3:6] = torch.randn((args_cli.num_envs, 3), dtype=torch.float32) * 1
    
    print("[INFO] Starting viewer loop. Press Ctrl+C or close viewer to exit.")
    print("[INFO] Robot will remain statifonary with zero actions (operational space control).")
    print("[INFO] Action space: 7 DOF (3D position + 3D rotation as axis-angle + gripper)")

    timer = Timer()
    timer.start()
    steps = 0

    close_step = 50

    def save_episode_video():
        """Save collected frames for current episode to video file."""
        nonlocal episode_frames, video_writer, episode_num
        if not episode_frames or video_output_dir is None:
            return
        
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        # Create video from collected frames
        if video_height is not None and video_width is not None:
            video_path = os.path.join(video_output_dir, f"tactile_episode_{episode_num:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined, misc]
            writer = cv2.VideoWriter(video_path, fourcc, args_cli.video_fps, (video_width, video_height))  # type: ignore[attr-defined]
            
            for frame in episode_frames:
                writer.write(frame)
            
            writer.release()
            print(f"[INFO] Saved video: {video_path} ({len(episode_frames)} frames)")
        
        episode_frames = []

    def collect_tactile_frame():
        """Collect a tactile frame from the current state."""
        nonlocal video_height, video_width
        unwrapped_env: TestEnv = env.unwrapped  # type: ignore[assignment]
        tactile_cam = cast(any, getattr(unwrapped_env, "_tactile_cam", None))
        if hasattr(unwrapped_env, "_tactile_cam") and tactile_cam is not None:
            data = tactile_cam.data
            if hasattr(data, "taxim_tactile") and data.taxim_tactile is not None:
                taxim = data.taxim_tactile
                try:
                    # Shape: (num_envs, height, width, 3) - RGB images
                    taxim_cpu = taxim.detach().cpu().numpy()
                    
                    # Take first environment's tactile image
                    if len(taxim_cpu.shape) == 4:
                        # Shape: (num_envs, H, W, 3)
                        tactile_frame = taxim_cpu[0]  # Take first env
                    elif len(taxim_cpu.shape) == 3:
                        # Shape: (num_envs, H, W) - grayscale, convert to RGB
                        tactile_frame = np.stack([taxim_cpu[0]] * 3, axis=-1)
                    else:
                        return None
                    
                    # Ensure values are in [0, 255] range
                    if tactile_frame.max() <= 1.0:
                        tactile_frame = (tactile_frame * 255.0).astype(np.uint8)
                    else:
                        tactile_frame = np.clip(tactile_frame, 0, 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    tactile_frame_bgr = cv2.cvtColor(tactile_frame, cv2.COLOR_RGB2BGR)
                    
                    # Set video dimensions if not set
                    if video_height is None or video_width is None:
                        video_height, video_width = tactile_frame_bgr.shape[:2]
                    
                    return tactile_frame_bgr
                except (RuntimeError, AttributeError, ValueError, TypeError) as e:
                    if args_cli.print_tactile:
                        print(f"[TACTILE] Error collecting frame: {e}")
                    return None
        return None

    try:
        while simulation_app.is_running():
            # Step environment with zero actions
            if steps < close_step:
                _, _, terminateds, truncateds, _ = env.step(down_actions)
            else:
                _, _, terminateds, truncateds, _ = env.step(close_actions)
            
            # Check for episode resets (when environment 0 is done)
            env_done = bool((terminateds | truncateds)[0])
            if env_done and video_output_dir:
                # Episode ended for environment 0
                # Save current episode video if we have frames
                if episode_frames:
                    save_episode_video()
                    episode_num += 1
                    episode_frames = []
            
            # Collect tactile frame for video recording
            if video_output_dir:
                tactile_frame = collect_tactile_frame()
                if tactile_frame is not None:
                    episode_frames.append(tactile_frame)

            # Print tactile info if requested
            if args_cli.print_tactile and steps % 100 == 0:
                unwrapped_env = env.unwrapped
                if hasattr(unwrapped_env, "_tactile_cam") and unwrapped_env._tactile_cam is not None:
                    data = unwrapped_env._tactile_cam.data
                    if hasattr(data, "taxim_tactile") and data.taxim_tactile is not None:
                        print(f"[TACTILE] Step {steps}: Shape={data.taxim_tactile.shape}, "
                              f"Range=[{data.taxim_tactile.min():.3f}, {data.taxim_tactile.max():.3f}]")

            # FPS logging
            steps += 1
            if steps % 100 == 0:
                elapsed = timer.time_elapsed
                fps = steps / elapsed
                print(f"[INFO] Step {steps}, FPS: {fps:.2f}")
            
            if args_cli.steps > 0 and steps >= args_cli.steps:
                break
    finally:
        # Save any remaining frames when exiting
        if video_output_dir and episode_frames:
            save_episode_video()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

