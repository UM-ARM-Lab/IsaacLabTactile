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
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Number of training epochs. When resuming from a checkpoint, this is additional epochs on top of the checkpoint epoch.",
)
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
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from tactile_obs_wrapper_factory import build_tactile_obs_wrapper

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Apply task-specific overrides from agent config (e.g., Factory peg insertion specific parameters)
    task_overrides = agent_cfg["params"].get("task_overrides", {})
    if task_overrides:
        use_full_rotation = task_overrides.get("use_full_rotation")
        if use_full_rotation is not None and hasattr(env_cfg, "ctrl"):
            env_cfg.ctrl.use_full_rotation = bool(use_full_rotation)
            print(f"[INFO] Setting use_full_rotation to {env_cfg.ctrl.use_full_rotation}")

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

        # Override friction domain randomization if specified
        if hasattr(env_cfg, "task"):
            if "gripper_peg_friction_randomization" in task_overrides:
                env_cfg.task.gripper_peg_friction_randomization = task_overrides["gripper_peg_friction_randomization"]
                print(f"[INFO] Setting gripper_peg_friction_randomization to {env_cfg.task.gripper_peg_friction_randomization}")
            if "gripper_peg_friction_range" in task_overrides:
                env_cfg.task.gripper_peg_friction_range = task_overrides["gripper_peg_friction_range"]
                print(f"[INFO] Setting gripper_peg_friction_range to {env_cfg.task.gripper_peg_friction_range}")
            if "fixed_asset_friction_randomization" in task_overrides:
                env_cfg.task.fixed_asset_friction_randomization = task_overrides["fixed_asset_friction_randomization"]
                print(
                    "[INFO] Setting fixed_asset_friction_randomization to "
                    f"{env_cfg.task.fixed_asset_friction_randomization}"
                )
            if "fixed_asset_friction_range" in task_overrides:
                env_cfg.task.fixed_asset_friction_range = task_overrides["fixed_asset_friction_range"]
                print(f"[INFO] Setting fixed_asset_friction_range to {env_cfg.task.fixed_asset_friction_range}")

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
            if "joint_friction_randomization" in task_overrides:
                env_cfg.task.joint_friction_randomization = task_overrides["joint_friction_randomization"]
                print(
                    "[INFO] Setting joint_friction_randomization to "
                    f"{env_cfg.task.joint_friction_randomization}"
                )
            if "joint_friction_range" in task_overrides:
                env_cfg.task.joint_friction_range = task_overrides["joint_friction_range"]
                print(f"[INFO] Setting joint_friction_range to {env_cfg.task.joint_friction_range}")
            if "action_threshold_randomization" in task_overrides:
                env_cfg.task.action_threshold_randomization = task_overrides["action_threshold_randomization"]
                print(
                    "[INFO] Setting action_threshold_randomization to "
                    f"{env_cfg.task.action_threshold_randomization}"
                )
            if "pos_threshold_noise_level" in task_overrides:
                env_cfg.task.pos_threshold_noise_level = float(task_overrides["pos_threshold_noise_level"])
                print(f"[INFO] Setting pos_threshold_noise_level to {env_cfg.task.pos_threshold_noise_level}")
            if "rot_threshold_noise_level" in task_overrides:
                env_cfg.task.rot_threshold_noise_level = float(task_overrides["rot_threshold_noise_level"])
                print(f"[INFO] Setting rot_threshold_noise_level to {env_cfg.task.rot_threshold_noise_level}")
            if hasattr(env_cfg, "task"):
                enable_contact_penalty = task_overrides.get("enable_contact_penalty", None)
                if enable_contact_penalty is not None:
                    if not bool(enable_contact_penalty):
                        env_cfg.task.contact_penalty_scale = 0.0
                        print("[INFO] Contact penalty disabled (enable_contact_penalty=false)")
                    elif (
                        "contact_penalty_scale" in task_overrides
                        and task_overrides["contact_penalty_scale"] is not None
                    ):
                        env_cfg.task.contact_penalty_scale = float(task_overrides["contact_penalty_scale"])
                        print(f"[INFO] Setting contact_penalty_scale to {env_cfg.task.contact_penalty_scale}")
                    else:
                        print(f"[INFO] Contact penalty enabled (scale={env_cfg.task.contact_penalty_scale})")
                elif "contact_penalty_scale" in task_overrides and task_overrides["contact_penalty_scale"] is not None:
                    env_cfg.task.contact_penalty_scale = float(task_overrides["contact_penalty_scale"])
                    print(f"[INFO] Setting contact_penalty_scale to {env_cfg.task.contact_penalty_scale}")
                if (
                    "contact_penalty_threshold_range" in task_overrides
                    and task_overrides["contact_penalty_threshold_range"] is not None
                ):
                    env_cfg.task.contact_penalty_threshold_range = task_overrides["contact_penalty_threshold_range"]
                    print(
                        f"[INFO] Setting contact_penalty_threshold_range to "
                        f"{env_cfg.task.contact_penalty_threshold_range}"
                    )

        # Override observation noise enable flag if specified
        obs_noise = task_overrides.get("obs_noise", None)
        if obs_noise is not None and hasattr(env_cfg, "obs_rand"):
            if isinstance(obs_noise, dict) and "enable_obs_noise" in obs_noise:
                env_cfg.obs_rand.enable_obs_noise = obs_noise["enable_obs_noise"]
                print(f"[INFO] Setting enable_obs_noise to {obs_noise['enable_obs_noise']}")
            else:
                print("[WARNING] obs_noise must be a dictionary with 'enable_obs_noise' key")

        if "fixed_asset_pos" in task_overrides and task_overrides["fixed_asset_pos"] is not None:
            if hasattr(env_cfg, "obs_rand"):
                env_cfg.obs_rand.fixed_asset_pos = list(task_overrides["fixed_asset_pos"])
                print(f"[INFO] Setting fixed_asset_pos to {env_cfg.obs_rand.fixed_asset_pos}")
            else:
                print("[WARNING] fixed_asset_pos override ignored: env has no obs_rand config")

        if "action_noise_std" in task_overrides:
            action_noise_std = float(task_overrides["action_noise_std"])
            if action_noise_std < 0.0:
                raise ValueError(f"action_noise_std must be non-negative, got {action_noise_std}")
            if hasattr(env_cfg, "action_pos_noise_std"):
                env_cfg.action_noise_model = None
                env_cfg.action_pos_noise_std = action_noise_std
                print(f"[INFO] Setting action_pos_noise_std to {action_noise_std} m")
            elif action_noise_std > 0.0:
                env_cfg.action_noise_model = NoiseModelCfg(
                    noise_cfg=GaussianNoiseCfg(mean=0.0, std=action_noise_std, operation="add"),
                )
                print(f"[INFO] Setting action_noise_std to {action_noise_std}")
            else:
                env_cfg.action_noise_model = None

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

    tactile_wrap_fn = build_tactile_obs_wrapper(env_cfg=env_cfg, task_overrides=task_overrides, rl_device=rl_device)

    # create isaac environment
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    
    env = base_env

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
