# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run for evaluation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import random
import time
import torch
from omegaconf import OmegaConf

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from tactile_transfer.model.point_latent_projection import load_point_latent_projection_from_checkpoint
from tactile_transfer.utils.rl_pointmae_wrapper import (
    PointMAEObsWrapper,
    load_pointmae_encoder_from_checkpoint,
)
from tactile_transfer.utils.rl_pointmae_sinkhorn_projection_wrapper import PointMAEObsWithSinkhornProjectionWrapper
from tactile_transfer.utils.rl_tactile_image_mae_wrapper import (
    TactileImageMAEObsWrapper,
    load_tactile_image_mae_encoder_from_checkpoint,
)

# Rectified-flow latent wrapper (tactile image <-> point latent)
from tactile_transfer.utils.rl_latent_ot_tactile_image_to_point_latent_wrapper import (
    TactileLatentFlowObsWrapper,
)

# Rectified-flow latent OT: tactile image -> pooled image latent -> point pooled latent
from tactile_transfer.model import (  # noqa: E402
    LatentNormalization,
    RectifiedFlowVelocityConfig,
    VelocityMLP,
)

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Apply task-specific overrides from agent config (e.g., Factory peg insertion specific parameters)
    task_overrides = agent_cfg["params"].get("task_overrides", {})
    if task_overrides:
        # Override tactile sensor calibration if specified
        if task_overrides.get("use_real_calib", False) and hasattr(env_cfg, "tactile_cam") and env_cfg.tactile_cam is not None:
            env_cfg.tactile_cam.calib_variant = "real"
        
        # Override gripper-peg friction if specified (ignored when gripper_peg_friction_randomization is True)
        gripper_peg_friction = task_overrides.get("gripper_peg_friction", None)
        if gripper_peg_friction is not None:
            if hasattr(env_cfg, "task") and hasattr(env_cfg.task, "gripper_peg_friction"):
                env_cfg.task.gripper_peg_friction = gripper_peg_friction
                print(f"[INFO] Setting gripper-peg friction to {gripper_peg_friction}")
            else:
                print("[WARNING] gripper_peg_friction parameter not available in this task configuration")

        # Override gripper-peg friction domain randomization if specified
        if hasattr(env_cfg, "task"):
            if "gripper_peg_friction_randomization" in task_overrides:
                env_cfg.task.gripper_peg_friction_randomization = task_overrides["gripper_peg_friction_randomization"]
                print(f"[INFO] Setting gripper_peg_friction_randomization to {env_cfg.task.gripper_peg_friction_randomization}")
            if "gripper_peg_friction_range" in task_overrides:
                env_cfg.task.gripper_peg_friction_range = task_overrides["gripper_peg_friction_range"]
                print(f"[INFO] Setting gripper_peg_friction_range to {env_cfg.task.gripper_peg_friction_range}")

        # Override observation noise enable flag if specified
        obs_noise = task_overrides.get("obs_noise", None)
        if obs_noise is not None and hasattr(env_cfg, "obs_rand"):
            if isinstance(obs_noise, dict) and "enable_obs_noise" in obs_noise:
                env_cfg.obs_rand.enable_obs_noise = obs_noise["enable_obs_noise"]
                print(f"[INFO] Setting enable_obs_noise to {obs_noise['enable_obs_noise']}")
            else:
                print("[WARNING] obs_noise must be a dictionary with 'enable_obs_noise' key")
        
        # Override include_held_asset_obs flag if specified
        include_held_asset_obs = task_overrides.get("include_held_asset_obs", None)
        if include_held_asset_obs is not None:
            # Initialize params structure if needed
            if env_cfg.params is None:
                env_cfg.params = OmegaConf.create({})
            if "env" not in env_cfg.params:
                env_cfg.params["env"] = OmegaConf.create({})
            env_cfg.params["env"]["include_held_asset_obs"] = include_held_asset_obs
            print(f"[INFO] Setting include_held_asset_obs to {include_held_asset_obs}")
            # Call update_env_params() again to apply the change to obs_order
            env_cfg.update_env_params()

    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    concate_state_groups = agent_cfg["params"]["env"].get("concate_state_groups", None)

    # Rectified-flow latent OT configuration (tactile image -> point latent)
    latent_ot_cfg = task_overrides.get("tactile_transfer_latent_ot", {})
    latent_ot_transfer_checkpoint = latent_ot_cfg.get("transfer_checkpoint")
    latent_ot_image_mae_checkpoint = latent_ot_cfg.get("image_mae_checkpoint")
    use_latent_ot = latent_ot_transfer_checkpoint is not None and latent_ot_transfer_checkpoint != "null"

    latent_ot_tactile_obs_key = latent_ot_cfg.get("tactile_obs_key", "tactile")
    latent_ot_euler_steps = int(latent_ot_cfg.get("euler_steps", 32))

    # Tactile-image MAE (direct image latent, matches training `train.py`)
    tactile_image_mae_cfg = task_overrides.get("tactile_image_mae", {})
    tactile_image_mae_checkpoint = tactile_image_mae_cfg.get("checkpoint")
    tactile_image_mae_obs_key = tactile_image_mae_cfg.get("tactile_obs_key", "tactile")
    use_tactile_image_mae = tactile_image_mae_checkpoint is not None and tactile_image_mae_checkpoint != "null"

    if use_tactile_image_mae:
        if hasattr(env_cfg, "enable_tactile_sensor"):
            env_cfg.enable_tactile_sensor = True

    inference_device = torch.device(rl_device if torch.cuda.is_available() else "cpu")
    # Latent-OT wrapper args are built only when `use_latent_ot=True`.
    latent_ot_wrapper_kwargs = None

    # Point-MAE tactile point cloud encoder configuration (standalone wrapper only).
    pointmae_wrapper_cfg = task_overrides.get("tactile_pointmae", {})
    pointmae_checkpoint = pointmae_wrapper_cfg.get("checkpoint")
    use_pointmae = pointmae_checkpoint is not None and pointmae_checkpoint != "null"
    # PointMAE-only optional projection MLP (applied in PointMAEObsWrapper).
    # Latent-OT has its own optional projection:
    # `tactile_transfer_latent_ot.point_latent_projection_checkpoint`.
    pointmae_sinkhorn_projection_ckpt = pointmae_wrapper_cfg.get("sinkhorn_projection_checkpoint")
    use_pointmae_sinkhorn_projection = (
        pointmae_sinkhorn_projection_ckpt is not None
        and str(pointmae_sinkhorn_projection_ckpt) != "null"
    )
    if use_pointmae_sinkhorn_projection and not use_pointmae:
        raise ValueError(
            "task_overrides.tactile_pointmae.sinkhorn_projection_checkpoint is set but "
            "tactile_pointmae.checkpoint is missing."
        )

    pointmae_model = None
    pointmae_sinkhorn_projection = None
    if use_pointmae:
        mae_device = torch.device(rl_device if torch.cuda.is_available() else "cpu")
        pointmae_model = load_pointmae_encoder_from_checkpoint(
            pointmae_checkpoint,
            mae_device,
            pointmae_wrapper_cfg,
        )
        if use_pointmae_sinkhorn_projection:
            pointmae_sinkhorn_projection = load_point_latent_projection_from_checkpoint(
                pointmae_sinkhorn_projection_ckpt,
                mae_device,
                expected_latent_dim=int(pointmae_model.cfg.embed_dim),
            )

        # Ensure env is configured to produce the observations PointMAE expects.
        if hasattr(env_cfg, "include_contact_forces"):
            env_cfg.include_contact_forces = True
        else:
            raise ValueError("include_contact_forces must be set to True in env_cfg")
        if hasattr(env_cfg, "include_tactile_pointclouds"):
            env_cfg.include_tactile_pointclouds = True
        else:
            raise ValueError("include_tactile_pointclouds must be set to True in env_cfg")

        print("[INFO] Point-MAE encoder ENABLED (play):")
        print(f"  - Checkpoint: {pointmae_checkpoint}")
        if use_pointmae_sinkhorn_projection:
            print(f"  - Sinkhorn point-latent projection: {pointmae_sinkhorn_projection_ckpt}")
        print(f"  - force_share_with_patches: {getattr(pointmae_model.cfg, 'force_share_with_patches', None)}")
        print(f"  - include_contact_forces: {getattr(env_cfg, 'include_contact_forces', None)}")
        print(f"  - include_tactile_pointclouds: {getattr(env_cfg, 'include_tactile_pointclouds', None)}")
    else:
        print("[INFO] Point-MAE encoder DISABLED - playing without Point-MAE tactile encoding")

    tactile_image_mae_model = None
    if use_tactile_image_mae:
        mae_device = torch.device(rl_device if torch.cuda.is_available() else "cpu")
        tactile_image_mae_model = load_tactile_image_mae_encoder_from_checkpoint(
            tactile_image_mae_checkpoint,
            mae_device,
        )
        print("[INFO] Tactile-image MAE encoder ENABLED (play):")
        print(f"  - Checkpoint: {tactile_image_mae_checkpoint}")
        print(f"  - tactile_obs_key: {tactile_image_mae_obs_key}")
        print(f"  - encoder_embed_dim: {tactile_image_mae_model.cfg.encoder_embed_dim}")
        print(f"  - enable_tactile_sensor: {getattr(env_cfg, 'enable_tactile_sensor', None)}")
    else:
        print("[INFO] Tactile-image MAE encoder DISABLED - playing without direct tactile-image MAE encoding")

    if use_latent_ot and use_pointmae:
        raise ValueError(
            "Both 'tactile_pointmae' and 'tactile_transfer_latent_ot' are enabled. "
            "Choose only one because both would append a latent token to vector observations."
        )
    if use_latent_ot and use_tactile_image_mae:
        raise ValueError(
            "Both 'tactile_transfer_latent_ot' and 'tactile_image_mae' are enabled. "
            "Choose only one because both append a tactile latent token to vector observations."
        )
    if use_pointmae and use_tactile_image_mae:
        raise ValueError(
            "Both 'tactile_pointmae' and 'tactile_image_mae' are enabled. "
            "Choose only one because both append latent embeddings to vector observations."
        )

    if use_latent_ot:
        # 1) Load rectified-flow modules from the latent OT checkpoint.
        transfer_ckpt_path = str(latent_ot_transfer_checkpoint)
        image_mae_ckpt_path = None
        transfer_payload = torch.load(transfer_ckpt_path, map_location="cpu")

        if "velocity" not in transfer_payload or "latent_normalization" not in transfer_payload:
            raise ValueError(
                "Latent OT transfer checkpoint is missing expected keys. "
                "Expected 'velocity' and 'latent_normalization'."
            )

        latent_dim = int(transfer_payload["latent_dim"])

        velocity_cfg_dict = transfer_payload.get("velocity_cfg", {})
        hidden_dims = velocity_cfg_dict.get("hidden_dims", (1024, 1024))
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        velocity_cfg = RectifiedFlowVelocityConfig(
            time_dim=int(velocity_cfg_dict.get("time_dim", 256)),
            proprio_dim=int(velocity_cfg_dict.get("proprio_dim", transfer_payload.get("proprio_dim", 0))),
            hidden_dims=tuple(hidden_dims),
        )

        velocity = VelocityMLP(latent_dim=latent_dim, cfg=velocity_cfg).to(inference_device)
        velocity.load_state_dict(transfer_payload["velocity"], strict=True)
        velocity.eval()
        transfer_prediction_target = str(transfer_payload.get("prediction_target", "velocity"))
        if transfer_prediction_target not in ("velocity", "x0"):
            raise ValueError(
                f"Unsupported transfer checkpoint prediction_target={transfer_prediction_target!r}. "
                "Expected 'velocity' or 'x0'."
            )

        # 2) Load the *source* latent encoder depending on checkpoint direction.
        latent_ot_direction = str(transfer_payload.get("direction", "image_to_pc"))
        if latent_ot_direction not in ("image_to_pc", "pc_to_image"):
            raise ValueError(
                f"Transfer checkpoint 'direction' must be 'image_to_pc' or 'pc_to_image', "
                f"got {latent_ot_direction!r}"
            )
        latent_norm = LatentNormalization.from_state_dict(transfer_payload["latent_normalization"]).to(inference_device)
        image_mae = None
        latent_ot_point_mae_model = None
        latent_ot_point_projection_model = None
        latent_ot_pc_gather_cfg = None

        if latent_ot_direction == "image_to_pc":
            if latent_ot_image_mae_checkpoint is None or str(latent_ot_image_mae_checkpoint) == "null":
                raise ValueError(
                    "Transfer checkpoint direction is image_to_pc; set "
                    "'image_mae_checkpoint' (or 'tactile_image_mae_checkpoint') under tactile_transfer_latent_ot."
                )
            image_mae_ckpt_path = str(latent_ot_image_mae_checkpoint)
            image_mae = load_tactile_image_mae_encoder_from_checkpoint(image_mae_ckpt_path, inference_device)
            img_latent_dim = int(image_mae.cfg.encoder_embed_dim)
            if img_latent_dim != latent_dim:
                raise ValueError(
                    f"TactileImageMAE encoder_embed_dim={img_latent_dim} != transfer latent_dim={latent_dim}."
                )
        else:
            ot_point_ckpt = latent_ot_cfg.get("point_mae_checkpoint")
            if ot_point_ckpt is None or str(ot_point_ckpt) == "null":
                raise ValueError(
                    "Transfer checkpoint direction is pc_to_image; set "
                    "'point_mae_checkpoint' under tactile_transfer_latent_ot."
                )
            point_mae_ckpt_path = str(ot_point_ckpt)
            pc_keys = latent_ot_cfg.get("pc_keys")
            force_keys = latent_ot_cfg.get("force_keys") or {}
            ot_point_projection_ckpt = (
                latent_ot_cfg.get("point_latent_projection_checkpoint")
            )
            if not pc_keys:
                raise ValueError(
                    "Transfer checkpoint direction is pc_to_image; set non-empty "
                    "pc_keys under tactile_transfer_latent_ot."
                )
            latent_ot_point_mae_model = load_pointmae_encoder_from_checkpoint(
                point_mae_ckpt_path,
                inference_device,
            )
            pc_embed_dim = int(latent_ot_point_mae_model.cfg.embed_dim)
            if pc_embed_dim != latent_dim:
                raise ValueError(
                    f"PointMAE embed_dim={pc_embed_dim} != transfer latent_dim={latent_dim}."
                )
            if ot_point_projection_ckpt is not None and str(ot_point_projection_ckpt) != "null":
                latent_ot_point_projection_model = load_point_latent_projection_from_checkpoint(
                    str(ot_point_projection_ckpt),
                    inference_device,
                    expected_latent_dim=latent_dim,
                )
            latent_ot_pc_gather_cfg = {"pc_keys": list(pc_keys), "force_keys": dict(force_keys)}

        # 3) Load proprio normalization stats from the checkpoint.
        if latent_ot_direction == "image_to_pc":
            latent_ot_proprio_src_mean = transfer_payload.get("proprio_img_mean")
            latent_ot_proprio_src_std = transfer_payload.get("proprio_img_std")
        else:
            latent_ot_proprio_src_mean = transfer_payload.get("proprio_pc_mean")
            latent_ot_proprio_src_std = transfer_payload.get("proprio_pc_std")
        if latent_ot_proprio_src_mean is None or latent_ot_proprio_src_std is None:
            # Fallback for paired-flow checkpoints that may only carry source-side stats.
            latent_ot_proprio_src_mean = transfer_payload.get("proprio_src_mean")
            latent_ot_proprio_src_std = transfer_payload.get("proprio_src_std")
        if latent_ot_proprio_src_mean is None or latent_ot_proprio_src_std is None:
            raise ValueError(
                "Transfer checkpoint is missing source proprio normalization stats. "
                "Re-export checkpoint with proprio statistics (proprio_img_mean/std and proprio_pc_mean/std)."
            )
        latent_ot_proprio_src_mean = latent_ot_proprio_src_mean.to(inference_device).float().reshape(-1)
        latent_ot_proprio_src_std = latent_ot_proprio_src_std.to(inference_device).float().reshape(-1)

        # Env must expose the correct tactile modality before gym.make (direction comes from checkpoint).
        if latent_ot_direction == "image_to_pc":
            if hasattr(env_cfg, "enable_tactile_sensor"):
                env_cfg.enable_tactile_sensor = True
        elif latent_ot_direction == "pc_to_image":
            if hasattr(env_cfg, "include_contact_forces"):
                env_cfg.include_contact_forces = True
            else:
                raise ValueError("include_contact_forces must be set to True in env_cfg for latent OT pc_to_image")
            if hasattr(env_cfg, "include_tactile_pointclouds"):
                env_cfg.include_tactile_pointclouds = True
            else:
                raise ValueError(
                    "include_tactile_pointclouds must be set to True in env_cfg for latent OT pc_to_image"
                )

        print("[INFO] Rectified-flow latent OT ENABLED (play):")
        print(f"  - transfer checkpoint: {transfer_ckpt_path}")
        if image_mae_ckpt_path is not None:
            print(f"  - image MAE checkpoint: {image_mae_ckpt_path}")
        print(f"  - direction (from transfer checkpoint): {latent_ot_direction}")
        print(f"  - euler_steps: {latent_ot_euler_steps}")
        print(f"  - prediction_target (ckpt): {transfer_prediction_target}")
        print(f"  - tactile_obs_key: {latent_ot_tactile_obs_key}")
        print(f"  - latent_dim: {latent_dim}")
        if latent_ot_direction == "pc_to_image" and latent_ot_pc_gather_cfg is not None:
            print(f"  - point MAE (flow source): {latent_ot_cfg.get('point_mae_checkpoint')}")
            print(f"  - pc_keys: {latent_ot_pc_gather_cfg.get('pc_keys')}")
            if latent_ot_point_projection_model is not None:
                print(
                    "  - point latent projection: "
                    f"{latent_ot_cfg.get('point_latent_projection_checkpoint')}"
                )

        latent_ot_wrapper_kwargs = dict(
            image_mae=image_mae,
            velocity=velocity,
            latent_norm=latent_norm,
            device=inference_device,
            latent_dim=latent_dim,
            tactile_obs_key=latent_ot_tactile_obs_key,
            euler_steps=latent_ot_euler_steps,
            prediction_target=transfer_prediction_target,
            direction=latent_ot_direction,
            point_mae=latent_ot_point_mae_model,
            point_latent_projection=latent_ot_point_projection_model,
            pc_gather_cfg=latent_ot_pc_gather_cfg,
            proprio_src_mean=latent_ot_proprio_src_mean,
            proprio_src_std=latent_ot_proprio_src_std,
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap with tactile encoding for policy conditioning
    if use_latent_ot:
        if latent_ot_wrapper_kwargs is None:
            raise RuntimeError("latent_ot_wrapper_kwargs must be set when use_latent_ot=True")
        direction = latent_ot_wrapper_kwargs["direction"]
        if direction == "image_to_pc":
            assert latent_ot_wrapper_kwargs["image_mae"] is not None, "image_mae must be loaded when direction=image_to_pc"
        elif direction == "pc_to_image":
            assert latent_ot_wrapper_kwargs["point_mae"] is not None, "point_mae must be loaded when direction=pc_to_image"
        env = TactileLatentFlowObsWrapper(env, **latent_ot_wrapper_kwargs)
    elif use_pointmae and pointmae_model is not None:
        mae_wrap_device = torch.device(rl_device if torch.cuda.is_available() else "cpu")
        if pointmae_sinkhorn_projection is not None:
            env = PointMAEObsWithSinkhornProjectionWrapper(
                env,
                pointmae_model,
                mae_wrap_device,
                pointmae_wrapper_cfg,
                pointmae_sinkhorn_projection,
            )
        else:
            env = PointMAEObsWrapper(
                env,
                pointmae_model,
                mae_wrap_device,
                pointmae_wrapper_cfg,
            )
    elif use_tactile_image_mae and tactile_image_mae_model is not None:
        env = TactileImageMAEObsWrapper(
            env,
            image_mae=tactile_image_mae_model,
            device=inference_device,
            tactile_obs_key=tactile_image_mae_obs_key,
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore[arg-type]

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups, concate_state_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if getattr(agent, "is_rnn", False):
        agent.init_rnn()
    
    # Episode tracking for evaluation
    if args_cli.num_episodes is not None:
        num_envs = env.unwrapped.num_envs
        episode_count = torch.zeros(num_envs, dtype=torch.int32, device=env.unwrapped.device)
        success_count = torch.zeros(num_envs, dtype=torch.int32, device=env.unwrapped.device)
        total_episodes_completed = 0
        print(f"[INFO] Running evaluation for {args_cli.num_episodes} episodes per environment ({num_envs} parallel environments)")
    
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, infos = env.step(actions)

            # perform operations for terminated episodes
            if torch.sum(dones) > 0:
                curr_successes = infos["curr_successes"]
                done_indices = dones.nonzero(as_tuple=False)
                success_count[done_indices] += curr_successes[done_indices].int()
                episode_count[done_indices] += 1
                total_episodes_completed = episode_count.sum().item()
                
                # Print progress
                print(f"[INFO] Episodes completed: {total_episodes_completed}/{args_cli.num_episodes * num_envs}")
        
                # Check if all environments have completed the required number of episodes
                if torch.all(episode_count >= args_cli.num_episodes):
                    break
    
                # reset rnn state for terminated episodes
                if getattr(agent, "is_rnn", False) and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # Report success rate if running evaluation
    if args_cli.num_episodes is not None:
        total_successes = success_count.sum().item()
        total_episodes = episode_count.sum().item()
        success_rate = (total_successes / total_episodes) * 100 if total_episodes > 0 else 0.0
        print("\n" + "="*60)
        print(f"[EVALUATION RESULTS]")
        print(f"Total episodes: {total_episodes}")
        print(f"Successful episodes: {total_successes}")
        print(f"Success rate: {success_rate:.2f}%")
        print("="*60 + "\n")

    # close the simulator
    env.unwrapped.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[misc]
    # close sim app
    simulation_app.close()
