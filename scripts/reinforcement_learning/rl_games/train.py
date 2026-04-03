# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
to run the training of peg insertion with tactile sensor readings as part of the input:
CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-PegInsert-Direct-v0 \
    --agent rl_games_ppo_tactile_cfg_entry_point \
    --enable_cameras \
    --num_envs 512 \
    --headless \
    env.enable_tactile_sensor=true \
    env.read_tactile_sensor=true \

to run one without tactile sensor readings:
CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-PegInsert-Direct-v0 \
    --agent rl_games_cfg_entry_point \ 
    --enable_cameras \
    --num_envs 512 \
    --headless


"""

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
from datetime import datetime
from pathlib import Path
from typing import Optional

import omni
import torch
from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
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
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from tactile_obs_wrapper_factory import build_tactile_obs_wrapper

# PLACEHOLDER: Extension template (do not remove this comment)


def load_transfer_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Load tactile transfer model checkpoint."""
    print(f"[INFO] Loading transfer checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint missing 'config'. Expect checkpoint from tactile transfer training.")
    return checkpoint


def create_transfer_model_from_config(config: dict, device: torch.device):
    """Create transfer model instance from config dictionary."""
    try:
        from tactile_transfer.model.unet import TactileUNet
    except ImportError as exc:
        raise ImportError(
            "Could not import TactileUNet from tactile_transfer. "
            "Please install tactile_transfer package: pip install -e /path/to/tactile_transfer"
        ) from exc
    
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("model_type", "unet")
    
    if model_type != "unet":
        raise ValueError(f"model_type must be 'unet', got '{model_type}'")
    
    obs_dim = config.get("obs_dim")
    action_dim = config.get("action_dim")
    tactile_image_shape = tuple(model_cfg.get("tactile_image_shape", [3, 64, 64]))
    tactile_encoder_base_channels = model_cfg.get("tactile_encoder_base_channels", 32)
    tactile_encoder_num_layers = model_cfg.get("tactile_encoder_num_layers", 4)
    obs_encoder_hidden_dims = model_cfg.get("obs_encoder_hidden_dims", [128, 64])
    latent_dim = model_cfg.get("latent_dim", 128)
    combine_method = model_cfg.get("combine_method", "concat")
    # Whether the transfer model was trained with the extended 5-tuple mode
    # (s_{t-1}, I_{t-1}, a_{t-1}, s_t, I_t).
    use_full_tuple = model_cfg.get("use_full_tuple", False)

    if obs_dim is None or action_dim is None:
        raise ValueError("Transfer config must include 'obs_dim' and 'action_dim'.")

    skip_connection_layers = model_cfg.get("skip_connection_layers", None)
    if skip_connection_layers is not None:
        skip_connection_layers = list(skip_connection_layers)
    
    model = TactileUNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tactile_image_shape=tactile_image_shape,
        tactile_encoder_base_channels=tactile_encoder_base_channels,
        tactile_encoder_num_layers=tactile_encoder_num_layers,
        obs_encoder_hidden_dims=obs_encoder_hidden_dims,
        latent_dim=latent_dim,
        combine_method=combine_method,
        skip_connection_layers=skip_connection_layers,
        use_full_tuple=use_full_tuple,
    )
    
    return model.to(device)


class TactileTransferWrapper:
    """Wrapper that intercepts observations and transfers them before passing to RL-Games.
    
    This wrapper takes a temporal tuple as input to the transfer model and stores
    the transferred observations back into the replay buffer.
    
    In original mode (use_full_tuple=False), the transfer model consumes:
        (s_{t-1}, a_{t-1}, s_t, I_t)
    and outputs either I_t (image-only) or (s_{t-1}, s_t, I_t) when
    decode_observations=True.
    
    In extended mode (use_full_tuple=True), the transfer model consumes:
        (s_{t-1}, I_{t-1}, a_{t-1}, s_t, I_t)
    and outputs either (I_{t-1}, I_t) or the full 5-tuple
        (s_{t-1}, I_{t-1}, a_{t-1}, s_t, I_t)
    when decode_observations=True.
    """
    
    def __init__(self, env, transfer_model, device, obs_dim, action_dim, decode_observations=False):
        self.env = env
        self.transfer_model = transfer_model
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.decode_observations = decode_observations
        self.prev_obs: Optional[torch.Tensor] = None
        self.prev_action: Optional[torch.Tensor] = None
        # Previous tactile image in the *environment* domain (before transfer).
        # Only used when the underlying transfer model expects I_{t-1}.
        self.prev_tactile: Optional[torch.Tensor] = None
    
    @property
    def unwrapped(self):
        """Returns the base environment (delegates to wrapped env's unwrapped)."""
        return self.env.unwrapped
    
    @property
    def single_observation_space(self):
        """Returns the single observation space (delegates to wrapped env)."""
        return self.env.unwrapped.single_observation_space
    
    @property
    def single_action_space(self):
        """Returns the single action space (delegates to wrapped env)."""
        return self.env.single_action_space
    
    @property
    def observation_space(self):
        """Returns the observation space (delegates to wrapped env)."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Returns the action space (delegates to wrapped env)."""
        return self.env.action_space
    
    @property
    def num_envs(self):
        """Returns the number of environments (delegates to wrapped env)."""
        return self.env.unwrapped.num_envs
    
    @property
    def render_mode(self):
        """Returns the render mode (delegates to wrapped env)."""
        return getattr(self.env, 'render_mode', None)
    
    def render(self):
        """Render the environment (delegates to wrapped env)."""
        return self.env.render()
    
    def reset(self):
        """Reset the environment and internal state."""
        obs_dict, extras = self.env.reset()
        self.prev_obs = None
        self.prev_action = None
        self.prev_tactile = None
        return obs_dict, extras
    
    def step(self, actions):
        """Step the environment and transfer observations before returning."""
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # Transfer observations before returning (these will be saved to replay buffer)
        obs_dict = self._transfer_obs(obs_dict, actions)
        return obs_dict, rew, terminated, truncated, extras
    
    def _transfer_obs(self, obs_dict: dict, actions: Optional[torch.Tensor]) -> dict:
        """Transfer observations using the transfer model.
        
        Takes (s_t-1, a_t-1, s_t, tactile_image_t) and outputs transferred (s_t, tactile_image_t).
        """
        if "tactile" not in obs_dict:
            raise ValueError("Tactile observations not found in obs_dict. Transfer requires tactile sensor.")
        
        tactile_original = obs_dict["tactile"]  # Shape: [num_envs, C, H, W], normalized to [-1, 1]
        num_envs = tactile_original.shape[0]
        
        # Get vector observations for transfer model
        if "vector_obs" in obs_dict:
            vector_obs = obs_dict["vector_obs"]
        elif "policy" in obs_dict:
            vector_obs = obs_dict["policy"]
        else:
            # Fallback: try to extract from obs_dict
            vector_obs = None
            for key in ["policy", "vector_obs"]:
                if key in obs_dict:
                    vector_obs = obs_dict[key]
                    break
        
        if vector_obs is None:
            raise ValueError("Could not find vector observations for transfer. Expected 'policy' or 'vector_obs' in obs_dict.")
        
        # Move to device for vectorized processing
        tactile_batch = tactile_original.to(self.device)  # [num_envs, C, H, W]
        obs_current_batch = vector_obs.to(self.device)  # [num_envs, obs_dim]
        
        # Prepare batched obs_t_minus_1 and action_t_minus_1
        # Use previous obs/action if available, otherwise use current obs/zero action
        if self.prev_obs is not None and self.prev_action is not None:
            assert self.prev_obs.shape[0] == num_envs, f"prev_obs shape {self.prev_obs.shape} != {num_envs}"
            obs_tm1_batch = self.prev_obs.to(self.device)  # [num_envs, obs_dim]
            action_tm1_batch = self.prev_action.to(self.device)  # [num_envs, action_dim]
        else:
            # First step: use current obs for both t-1 and t
            obs_tm1_batch = obs_current_batch.clone()
            action_tm1_batch = actions.to(self.device) if actions is not None else torch.zeros(
                (num_envs, self.action_dim), device=self.device
            )
        
        # Validate dimensions
        if obs_tm1_batch.shape != (num_envs, self.obs_dim):
            raise ValueError(f"obs_tm1_batch shape {obs_tm1_batch.shape} != ({num_envs}, {self.obs_dim})")
        if obs_current_batch.shape != (num_envs, self.obs_dim):
            raise ValueError(f"obs_current_batch shape {obs_current_batch.shape} != ({num_envs}, {self.obs_dim})")
        if action_tm1_batch.shape != (num_envs, self.action_dim):
            raise ValueError(f"action_tm1_batch shape {action_tm1_batch.shape} != ({num_envs}, {self.action_dim})")
        
        # Prepare tactile_t_minus_1 batch if the transfer model expects full tuples
        if getattr(self.transfer_model, "use_full_tuple", False):
            if self.prev_tactile is not None:
                assert self.prev_tactile.shape[0] == num_envs, f"prev_tactile shape {self.prev_tactile.shape} != {num_envs}"
                tactile_tm1_batch = self.prev_tactile.to(self.device)
            else:
                # First step: use current tactile as I_{t-1}
                tactile_tm1_batch = tactile_batch.clone()
        else:
            tactile_tm1_batch = None
        
        # Vectorized transfer: process all environments at once
        with torch.inference_mode():
            if tactile_tm1_batch is not None:
                transfer_output = self.transfer_model.forward(
                    obs_tm1_batch,
                    action_tm1_batch,
                    obs_current_batch,
                    tactile_batch,
                    tactile_images_t_minus_1=tactile_tm1_batch,
                )
            else:
                transfer_output = self.transfer_model.forward(
                    obs_tm1_batch,
                    action_tm1_batch,
                    obs_current_batch,
                    tactile_batch,
                )
            
            # Handle output - generator always returns consistent tuple structure
            # regardless of decode_observations flag
            if getattr(self.transfer_model, "use_full_tuple", False):
                # Extended mode: always returns 5-tuple (s_{t-1}, I_{t-1}, a_{t-1}, s_t, I_t)
                (
                    _gen_obs_t_minus_1_batch,
                    _gen_tactile_t_minus_1_batch,
                    _gen_action_t_minus_1_batch,
                    decoded_obs_t_batch,
                    tactile_transferred,
                ) = transfer_output
                # Only use decoded observations if decode_observations is enabled
                if not self.decode_observations:
                    decoded_obs_t_batch = None
            else:
                # Original mode: always returns 3-tuple (s_{t-1}, s_t, I_t)
                _gen_obs_t_minus_1_batch, decoded_obs_t_batch, tactile_transferred = transfer_output
                # Only use decoded observations if decode_observations is enabled
                if not self.decode_observations:
                    decoded_obs_t_batch = None
        
        # Clamp to ensure values are in [-1, 1] range
        tactile_transferred = torch.clamp(tactile_transferred, -1.0, 1.0)
        
        # Replace original tactile with transferred version (keep on same device)
        obs_dict["tactile"] = tactile_transferred.to(obs_dict["tactile"].device)
        
        # Replace vector observations with decoded observations if decode_observations is enabled
        if self.decode_observations and decoded_obs_t_batch is not None:
            # Replace vector observations in obs_dict
            if "vector_obs" in obs_dict:
                obs_dict["vector_obs"] = decoded_obs_t_batch.to(obs_dict["vector_obs"].device)
            elif "policy" in obs_dict:
                obs_dict["policy"] = decoded_obs_t_batch.to(obs_dict["policy"].device)
            else:
                # Try to find and replace any vector observation key
                for key in ["policy", "vector_obs"]:
                    if key in obs_dict:
                        obs_dict[key] = decoded_obs_t_batch.to(obs_dict[key].device)
                        break
        
        # Update previous obs/action/tactile for next step (store batched versions)
        # Always track the *original* environment observations here for the transfer model.
        if actions is not None:
            assert actions.shape[0] == num_envs, f"actions shape {actions.shape} != {num_envs}"
            self.prev_action = actions.detach().clone()
        
        # `prev_obs` should track the original (source-domain) observations, not the
        # transferred/decoded ones. The decoded observations are only used to update
        # `obs_dict` above for the policy, while the transfer model's temporal inputs
        # (s_{t-1}, s_t) must remain in the original observation domain.
        if vector_obs is not None:
            self.prev_obs = vector_obs.detach().clone()
        
        # Track previous *original* tactile observations (environment domain)
        self.prev_tactile = tactile_original.detach().clone()
        
        return obs_dict
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
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
    
    # update agent device configuration to match environment device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg:
        if agent_cfg["pbt"]["directory"] == ".":
            log_root_path = os.path.abspath(log_root_path)
        else:
            log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    experiment_name = datetime.now().strftime("%m%d%H%M%S_")+args_cli.task if args_cli.wandb_name is None else args_cli.wandb_name
    log_dir = experiment_name
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    concate_state_groups = agent_cfg["params"]["env"].get("concate_state_groups", None)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Load tactile transfer and Point-MAE configurations from agent config
    task_overrides = agent_cfg["params"].get("task_overrides", {})

    # Tactile transfer (image/domain transfer) configuration
    tactile_transfer_cfg = task_overrides.get("tactile_transfer", {})
    transfer_checkpoint = tactile_transfer_cfg.get("transfer_checkpoint")
    transfer_direction = tactile_transfer_cfg.get("transfer_direction", "A_to_B")
    decode_observations = tactile_transfer_cfg.get("decode_observations", False)

    transfer_model = None
    obs_dim = None
    action_dim = None
    use_tactile_transfer = transfer_checkpoint is not None and transfer_checkpoint != "null"

    if use_tactile_transfer:
        transfer_checkpoint_path = Path(transfer_checkpoint)
        if not transfer_checkpoint_path.exists():
            raise FileNotFoundError(f"Transfer checkpoint not found: {transfer_checkpoint_path}")
        
        device = torch.device(rl_device if torch.cuda.is_available() else "cpu")
        checkpoint = load_transfer_checkpoint(transfer_checkpoint_path, device)
        config = checkpoint["config"]

        # Create both models (A and B)
        model_A = create_transfer_model_from_config(config, device)
        model_B = create_transfer_model_from_config(config, device)
        model_A.load_state_dict(checkpoint["model_A_state_dict"])
        model_B.load_state_dict(checkpoint["model_B_state_dict"])
        model_A.eval()
        model_B.eval()

        # Select model based on transfer direction
        if transfer_direction == "A_to_B":
            transfer_model = model_A  # model_A does A->B transfer
        elif transfer_direction == "B_to_A":
            transfer_model = model_B  # model_B does B->A transfer
        else:
            raise ValueError(f"Invalid transfer_direction: {transfer_direction}. Must be 'A_to_B' or 'B_to_A'")

        obs_dim = config.get("obs_dim")
        action_dim = config.get("action_dim")

        print("[INFO] Tactile transfer ENABLED:")
        print(f"  - Transfer checkpoint: {transfer_checkpoint_path}")
        print(f"  - Transfer direction: {transfer_direction}")
        print(f"  - Decode observations: {decode_observations}")
        print(f"  - obs_dim: {obs_dim}, action_dim: {action_dim}")
        print("  - Observations will be transferred before being saved to replay buffer")
    else:
        print("[INFO] Tactile transfer DISABLED - training without transfer (observations saved as-is to replay buffer)")

    tactile_wrap_fn = build_tactile_obs_wrapper(env_cfg=env_cfg, task_overrides=task_overrides, rl_device=rl_device)

    # create isaac environment
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    
    env = base_env

    # Wrap with tactile transfer wrapper if transfer is enabled
    if use_tactile_transfer and transfer_model is not None:
        env = TactileTransferWrapper(
            env,
            transfer_model,
            torch.device(rl_device if torch.cuda.is_available() else "cpu"),
            obs_dim,
            action_dim,
            decode_observations=decode_observations,
        )

    # Wrap with tactile encoding for policy conditioning (config-driven)
    env = tactile_wrap_fn(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups, concate_state_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(IsaacAlgoObserver())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
