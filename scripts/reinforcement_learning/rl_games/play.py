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
    help="When --checkpoint is not set, load the latest checkpoint from the experiment log directory.",
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

from tactile_obs_wrapper_factory import build_tactile_obs_wrapper

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

        elastomer_stiffness = task_overrides.get("elastomer_stiffness", None)
        if elastomer_stiffness is not None:
            elastomer_stiffness = float(elastomer_stiffness)
            if hasattr(env_cfg, "tactile_cam") and env_cfg.tactile_cam is not None:
                env_cfg.tactile_cam.compliance_stiffness = elastomer_stiffness
            if hasattr(env_cfg, "tactile_cam_right") and env_cfg.tactile_cam_right is not None:
                env_cfg.tactile_cam_right.compliance_stiffness = elastomer_stiffness
            print(f"[INFO] Setting elastomer compliance_stiffness to {elastomer_stiffness}")

        gripper_kp = task_overrides.get("gripper_kp", None)
        gripper_kd = task_overrides.get("gripper_kd", None)
        if gripper_kp is not None or gripper_kd is not None:
            if hasattr(env_cfg, "robot") and "panda_hand" in env_cfg.robot.actuators:
                if gripper_kp is not None:
                    env_cfg.robot.actuators["panda_hand"].stiffness = float(gripper_kp)
                    print(f"[INFO] Setting gripper_kp (panda_hand stiffness) to {gripper_kp}")
                if gripper_kd is not None:
                    env_cfg.robot.actuators["panda_hand"].damping = float(gripper_kd)
                    print(f"[INFO] Setting gripper_kd (panda_hand damping) to {gripper_kd}")

        if hasattr(env_cfg, "task"):
            if "arm_control_gain_randomization" in task_overrides:
                env_cfg.task.arm_control_gain_randomization = task_overrides["arm_control_gain_randomization"]
                print(
                    "[INFO] Setting arm_control_gain_randomization to "
                    f"{env_cfg.task.arm_control_gain_randomization}"
                )
            if "arm_kp_scale_range" in task_overrides:
                env_cfg.task.arm_kp_scale_range = task_overrides["arm_kp_scale_range"]
                print(f"[INFO] Setting arm_kp_scale_range to {env_cfg.task.arm_kp_scale_range}")
            if "arm_kd_scale_range" in task_overrides:
                env_cfg.task.arm_kd_scale_range = task_overrides["arm_kd_scale_range"]
                print(f"[INFO] Setting arm_kd_scale_range to {env_cfg.task.arm_kd_scale_range}")
            if "gripper_kp_kd_randomization" in task_overrides:
                env_cfg.task.gripper_kp_kd_randomization = task_overrides["gripper_kp_kd_randomization"]
                print(
                    "[INFO] Setting gripper_kp_kd_randomization to "
                    f"{env_cfg.task.gripper_kp_kd_randomization}"
                )
            if "gripper_kp_scale_range" in task_overrides:
                env_cfg.task.gripper_kp_scale_range = task_overrides["gripper_kp_scale_range"]
                print(f"[INFO] Setting gripper_kp_scale_range to {env_cfg.task.gripper_kp_scale_range}")
            if "gripper_kd_scale_range" in task_overrides:
                env_cfg.task.gripper_kd_scale_range = task_overrides["gripper_kd_scale_range"]
                print(f"[INFO] Setting gripper_kd_scale_range to {env_cfg.task.gripper_kd_scale_range}")

        # Override observation noise enable flag if specified
        obs_noise = task_overrides.get("obs_noise", None)
        if obs_noise is not None and hasattr(env_cfg, "obs_rand"):
            if isinstance(obs_noise, dict) and "enable_obs_noise" in obs_noise:
                env_cfg.obs_rand.enable_obs_noise = obs_noise["enable_obs_noise"]
                print(f"[INFO] Setting enable_obs_noise to {obs_noise['enable_obs_noise']}")
            else:
                print("[WARNING] obs_noise must be a dictionary with 'enable_obs_noise' key")

        tactile_pc_dr = task_overrides.get("tactile_pointcloud_dr", None)
        if isinstance(tactile_pc_dr, dict):
            if "held_pos_noise_std" in tactile_pc_dr and hasattr(env_cfg, "tactile_pointcloud_held_pos_noise_std"):
                env_cfg.tactile_pointcloud_held_pos_noise_std = float(tactile_pc_dr["held_pos_noise_std"])
                print(
                    "[INFO] Setting tactile_pointcloud_held_pos_noise_std to "
                    f"{env_cfg.tactile_pointcloud_held_pos_noise_std}"
                )
            if "held_rot_noise_std" in tactile_pc_dr and hasattr(env_cfg, "tactile_pointcloud_held_rot_noise_std"):
                env_cfg.tactile_pointcloud_held_rot_noise_std = float(tactile_pc_dr["held_rot_noise_std"])
                print(
                    "[INFO] Setting tactile_pointcloud_held_rot_noise_std to "
                    f"{env_cfg.tactile_pointcloud_held_rot_noise_std}"
                )
        
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

        # Override include_prev_actions flag if specified
        include_prev_actions = task_overrides.get("include_prev_actions", None)
        if include_prev_actions is not None:
            # Initialize params structure if needed
            if env_cfg.params is None:
                env_cfg.params = OmegaConf.create({})
            if "env" not in env_cfg.params:
                env_cfg.params["env"] = OmegaConf.create({})
            env_cfg.params["env"]["include_prev_actions"] = include_prev_actions
            print(f"[INFO] Setting include_prev_actions to {include_prev_actions}")
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
    # resolve policy checkpoint (optional)
    resume_path = None
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.use_last_checkpoint:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        resume_path = get_checkpoint_path(log_root_path, run_dir, ".*", other_dirs=["nn"])
    else:
        print(
            "[INFO] No policy checkpoint provided; rolling out with a randomly initialized policy "
            "(stochastic actions)."
        )

    load_policy_checkpoint = resume_path is not None
    log_dir = os.path.dirname(os.path.dirname(resume_path)) if load_policy_checkpoint else log_root_path

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    concate_state_groups = agent_cfg["params"]["env"].get("concate_state_groups", None)

    tactile_wrap_fn = build_tactile_obs_wrapper(env_cfg=env_cfg, task_overrides=task_overrides, rl_device=rl_device)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap with tactile encoding for policy conditioning (config-driven)
    env = tactile_wrap_fn(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
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

    agent_cfg["params"]["load_checkpoint"] = load_policy_checkpoint
    if load_policy_checkpoint:
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    if load_policy_checkpoint:
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
            is_deterministic = agent.is_deterministic if load_policy_checkpoint else False
            actions = agent.get_action(obs, is_deterministic=is_deterministic)
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
