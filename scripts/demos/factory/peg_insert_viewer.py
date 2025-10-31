#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
# noqa: SLF001

"""
Viewer demo for Factory Peg Insertion without tactile sensing.

This script:
 - Instantiates the Isaac-Factory-PegInsert-Direct-v0 task
 - Steps the environment with zero actions
 - Visualizes the scene via the Isaac Sim viewer

Usage:
    python scripts/demos/factory/peg_insert_viewer.py \
        --num_envs 16

Notes:
 - AppLauncher must be called first to set up Omniverse environment
 - Use --headless in AppLauncher args to disable the viewer
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Factory Peg Insertion viewer (no tactile)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 = run forever)")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
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
    # Create a config with tactile disabled
    # Pass minimal params to avoid None access inside __post_init__
    env_cfg = FactoryTaskPegInsertCfg(params=OmegaConf.create({"env": {}}))
    env_cfg.enable_tactile_sensor = False
    env_cfg.read_tactile_sensor = False
    env_cfg.enable_obs_camera = False
    env_cfg.use_compliant_gripper = True
    env_cfg.use_gelsight_finger = False
    env_cfg.scene.num_envs = args_cli.num_envs

    # Update simulation device
    if getattr(args_cli, "device", None) == "cpu":
        env_cfg.sim.device = "cpu"
    elif getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    # Create environment via gymnasium registry
    print("[INFO] Creating Factory Peg Insert environment (no tactile)...")
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


