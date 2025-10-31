#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
# noqa: SLF001

"""
Viewer demo for Factory Peg Insertion with TacSL visuo-tactile sensing.

This script:
 - Instantiates the Isaac-Factory-PegInsert-Direct-v0 task
 - Enables the tactile sensor and reading of tactile images
 - Steps the environment with zero actions
 - Visualizes the scene via the Isaac Sim viewer

Usage:
    python scripts/demos/factory/peg_insert_tactile_viewer.py \
        --num_envs 16 --enable_cameras --print_tactile

Notes:
 - AppLauncher must be called first to set up Omniverse environment
 - Use --enable_cameras to render the scene
 - Tactile images are exposed via env.unwrapped._tactile_cam.data.taxim_tactile when enabled
"""

import argparse
import numpy as np
from typing import cast

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Factory Peg Insertion viewer with tactile sensor")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 = run forever)")
parser.add_argument("--print_tactile", action="store_true", help="Print basic tactile stats each step")
# parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device")

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


def main():
    # Create a modified config that enables tactile sensing
    # Pass a minimal params to avoid None access inside __post_init__
    env_cfg = FactoryTaskPegInsertCfg(params=OmegaConf.create({"env": {}}))
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
    print("[INFO] Creating Factory Peg Insert environment with tactile sensor...")
    env: FactoryEnv = gym.make(
        "Isaac-Factory-PegInsert-Direct-v0",
        cfg=env_cfg,
    )
    print("[INFO] Environment created successfully.")

    # Reset once before stepping
    _ = env.reset()
    print("[INFO] Environment reset.")

    # Zero action loop
    action_dim = env.unwrapped.action_size if hasattr(env.unwrapped, "action_size") else 6
    zero_actions = np.zeros((args_cli.num_envs, action_dim), dtype=np.float32)

    print("[INFO] Starting viewer loop. Press Ctrl+C or close viewer to exit.")

    timer = Timer()
    timer.start()
    steps = 0

    try:
        while simulation_app.is_running():
            # Step environment with zero actions
            _ = env.step(zero_actions)

            # Optional tactile printout
            if args_cli.print_tactile:
                unwrapped_env: FactoryEnv = env.unwrapped  # type: ignore[assignment]
                # Access protected member intentionally for demo introspection
                tactile_cam = cast(any, getattr(unwrapped_env, "_tactile_cam", None))  # noqa: SLF001
                if hasattr(unwrapped_env, "_tactile_cam") and tactile_cam is not None:
                    data = tactile_cam.data  # noqa: SLF001
                    if hasattr(data, "taxim_tactile") and data.taxim_tactile is not None:
                        # Shape: [num_envs, H, W] (depth) or [num_envs, C, H, W] depending on config
                        taxim = data.taxim_tactile
                        try:
                            taxim_cpu = taxim.detach().cpu().numpy()
                            print(
                                f"[TACTILE] shape={taxim_cpu.shape} min={taxim_cpu.min():.4f} "
                                f"max={taxim_cpu.max():.4f}"
                            )
                        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
                            print(f"[TACTILE] available (tensor), error on access: {e}")

            # FPS logging
            steps += 1
            if steps % 120 == 0:
                fps = timer.average_fps
                print(f"[INFO] Average FPS: {fps:.1f} (steps: {steps})")

            if args_cli.steps > 0 and steps >= args_cli.steps:
                break

    except KeyboardInterrupt:
        print("[INFO] Exiting on user interrupt")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()


