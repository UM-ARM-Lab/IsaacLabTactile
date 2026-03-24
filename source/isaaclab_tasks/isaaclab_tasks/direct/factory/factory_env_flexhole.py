# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory environment with flexible hole sizes (mixed large/regular environments)."""

from typing import Dict, Union

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import factory_utils
from .factory_env import FactoryEnv
from .factory_env_cfg import (
    FactoryTaskBaseFlexHoleCfg,
    FactoryTaskNutThreadFlexHoleCfg,
    FactoryTaskPegInsertFlexHoleCfg,
)


class FactoryFlexHoleEnv(FactoryEnv):
    """Factory environment with sim/real split and mixed hole sizes.

    Index layout: [sim][real]
    - sim: Scaled hole (easier clearance, for sim-to-real transfer)
    - real: Regular hole (tight clearance, mimics real hardware)

    The environment does not know about "training" vs "validation" - that's
    context-dependent. It only needs to know about "sim" vs "real" hole sizes.
    Both success rates (success_sim, success_real) are logged to wandb.
    """

    cfg: FactoryTaskBaseFlexHoleCfg

    def __init__(self, cfg: FactoryTaskBaseFlexHoleCfg, render_mode: str | None = None, **kwargs):
        # Parse sim/real counts from flat config

        # Validate total matches scene.num_envs
        self._verify_num_envs(cfg)

        # Store scale and budget for later use
        self._sim_fixed_asset_scale = cfg.sim_fixed_asset_scale
        self._real_fixed_asset_scale = cfg.real_fixed_asset_scale

        # Set the params
        self.num_train_sim = cfg.num_train_sim
        self.num_train_real = cfg.num_train_real
        self.num_val_sim = cfg.num_val_sim
        self.num_val_real = cfg.num_val_real
        # No need for self.num_envs, since already resolved and auto-set in super()
        # self.num_envs = self.num_train_sim + self.num_train_real + self.num_val_sim + self.num_val_real
        self.num_train = self.num_train_sim + self.num_train_real
        self.num_eval = self.num_val_sim + self.num_val_real
        self.randomize_partition = cfg.randomize_partition

        # Budget tracking (episode-based)
        self.sim_budget = cfg.sim_budget
        self.real_budget = cfg.real_budget
        self.sim_budget_used = 0.0
        self.real_budget_used = 0.0
        self.total_budget_used = 0.0

        # Pre-adjust observation_space for 6D quaternion representation
        # Each quaternion field (4D) becomes 6D, adding 2 dimensions per field
        num_quat_fields_obs = sum(1 for obs in cfg.obs_order if obs.endswith("_quat"))
        num_quat_fields_state = sum(1 for s in cfg.state_order if s.endswith("_quat"))
        self._obs_quat_adjustment = num_quat_fields_obs * 2
        self._state_quat_adjustment = num_quat_fields_state * 2

        super().__init__(cfg, render_mode, **kwargs)

        # When self.device is available after super(), run this
        self._init_partitions()
        self._init_scales()
        # Apply per-environment hole scaling via USD API
        self._apply_flex_hole_scales()

        # Adjust the computed observation_space and state_space for 6D quaternions
        self.cfg.observation_space += self._obs_quat_adjustment
        self.cfg.state_space += self._state_quat_adjustment

        # Update the gym observation space Dict to match new dimensions
        import numpy as np
        from gymnasium import spaces

        new_obs_dim = self.cfg.observation_space
        new_state_dim = self.cfg.state_space
        self.single_observation_space["policy"] = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,))
        self.single_observation_space["critic"] = spaces.Box(low=-np.inf, high=np.inf, shape=(new_state_dim,))

    @classmethod
    def _verify_num_envs(cls, cfg):
        actual_total_envs = cfg.scene.num_envs
        expected_total_envs = cfg.num_train_sim + cfg.num_train_real + cfg.num_val_sim + cfg.num_val_real
        if actual_total_envs != expected_total_envs:
            raise ValueError(
                f"Total num_envs ({actual_total_envs}) must equal sum of sim/real train/eval envs ({expected_total_envs})"
            )
        return True

    def _init_partitions(self):
        # Index slices for sim/real
        train_ids = torch.arange(self.num_train, device=self.device)
        self.idx_train = torch.randperm(self.num_train, device=self.device) if self.randomize_partition else train_ids
        self.idx_train_real = self.idx_train[: self.num_train_real]
        self.idx_train_sim = self.idx_train[self.num_train_real :]

        # self.idx_train = slice(0, self.num_train)
        self.idx_val = torch.arange(self.num_train, self.num_train + self.num_eval, device=self.device)
        self.idx_val_real = self.idx_val[: self.num_val_real]
        self.idx_val_sim = self.idx_val[self.num_val_real :]

        # Compute sim indices and real indices for easy access
        self.idx_sim = torch.cat([self.idx_train_sim, self.idx_val_sim])
        self.idx_real = torch.cat([self.idx_train_real, self.idx_val_real])
        self._avail_slice_keys = set(["train", "val", "train_sim", "train_real", "val_sim", "val_real"])

    def _init_scales(self):
        """Initialize per-environment hole scale multipliers and compute corresponding XY success tolerances."""
        # Create scale multiplier tensor
        scales = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        scales[self.idx_real] = self._real_fixed_asset_scale
        scales[self.idx_sim] = self._sim_fixed_asset_scale
        self.asset_scale_multipliers = scales

        # Compute per-environment XY success tolerance.
        # When the hole is scaled UP (scale > 1), the peg has more room, so we loosen the
        # success criterion proportionally. When the bolt is scaled DOWN (scale < 1), the
        # success criterion is kept at the base tolerance (same as real).
        #   xy_tolerance = base_tolerance + (fixed_diameter * max(scale - 1, 0)) / 2
        base_xy_tolerance = 0.0025 * torch.ones(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )  # Original hardcoded tolerance
        # For peg flexhole task
        if self.cfg_task.name == "peg_insert":
            fixed_diameter = 0.009  # Inner diameter of the Hole8mm asset (9mm); cfg.diameter is the nominal peg size
            scale_increase = torch.clamp(self.asset_scale_multipliers - 1.0, min=0.0)
            self.xy_success_tolerance = base_xy_tolerance + (fixed_diameter * scale_increase) / 2

    def _apply_flex_hole_scales(self):
        """
        Apply per-environment hole scaling via USD API after clone_environments().
        """
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf

        stage = get_current_stage()
        # Apply both
        real_gp = (self.idx_real.tolist(), self._real_fixed_asset_scale)
        sim_gp = (self.idx_sim.tolist(), self._sim_fixed_asset_scale)
        for idx_list, scale in [real_gp, sim_gp]:
            for i in idx_list:
                fixed_asset = stage.GetPrimAtPath(f"/World/envs/env_{i}/FixedAsset")
                fixed_asset.GetAttribute("xformOp:scale").Set(Gf.Vec3f(scale, scale, 1.0))

    def _setup_scene(self):
        """Initialize simulation scene with mixed hole sizes using MultiAssetSpawnerCfg."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Spawn table
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        # Spawn assets
        self._robot = Articulation(self.cfg.robot)
        # self._fixed_asset = Articulation(self.cfg_task.fixed_assets)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        # Handle gear mesh task assets
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        # Register with scene
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # Copy environment
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add observation camera if enabled
        if self.cfg.use_obs_camera:
            self._obs_camera = TiledCamera(self.cfg.obs_camera_cfg)
            self.scene.sensors["obs_camera"] = self._obs_camera

        # Add contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

    @staticmethod
    def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion (w,x,y,z) to 6D rotation representation.

        The 6D representation uses the first two columns of the rotation matrix,
        which is continuous and avoids gimbal lock issues.

        Args:
            quat: Quaternion tensor of shape (..., 4) in (w, x, y, z) order (IsaacLab convention)

        Returns:
            6D rotation tensor of shape (..., 6)
        """
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # First column of rotation matrix
        v1 = 1 - 2 * y * y - 2 * z * z
        v2 = 2 * x * y + 2 * z * w
        v3 = 2 * x * z - 2 * y * w

        # Second column of rotation matrix
        v4 = 2 * x * y - 2 * z * w
        v5 = 1 - 2 * x * x - 2 * z * z
        v6 = 2 * y * z + 2 * x * w

        return torch.stack([v1, v2, v3, v4, v5, v6], dim=-1)

    def _get_observations(self):
        """Get observations with 6D quaternion representation.

        Returns all environments' observations.
        """
        obs_dict, state_dict, collect_dict = super()._get_factory_obs_state_dict()
        # Replace quaternion keys with 6D representation
        for d in [obs_dict, state_dict, collect_dict]:
            for key in d.keys():
                if key.endswith("_quat"):
                    d[key] = self.quat_to_6d(d[key])

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        # Store collection observations for data collection
        self.collect_obs = torch.cat([collect_dict[key] for key in collect_dict.keys()], dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}

    def get_slice_keys(self):
        """Get available slice keys for slicing observations/states according to sim/real partition."""
        return self._avail_slice_keys

    def slice_for_envs(self, item: Union[torch.Tensor, Dict], key: str) -> Union[torch.Tensor, Dict]:
        """
        Slice the given item (tensor or dict of tensors) according to the sim/real partition for the specified key.
        """
        assert key in self._avail_slice_keys, f"Invalid key: {key}"
        partition_key = getattr(self, f"idx_{key}")  # e.g. key="train_sim" -> self.idx_train_sim

        # If tensor, slice directly. If dict, apply slicing to each value (used for collect_obs dict during data collection).
        if isinstance(item, torch.Tensor):
            assert item.shape[0] == self.num_envs, (
                f"Expected first dimension to be num_envs ({self.num_envs}), got {item.shape[0]}"
            )
            return item[partition_key]

        if isinstance(item, dict):
            sliced_dict = {}
            for k, v in item.items():
                assert isinstance(v, torch.Tensor), f"Expected dict values to be tensors, got {type(v)} for key {k}"
                if v.shape[0] != self.num_envs:
                    sliced_dict[k] = v  # If not env-batched, return as is (e.g. scalar values)
                else:  # Otherwise slice along env dimension
                    sliced_dict[k] = v[partition_key]
            return sliced_dict

        assert False, f"Expected item to be either torch.Tensor or dict, got {type(item)}"

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask with per-environment XY tolerance based on hole scale.

        Uses affine tolerance: xy_tol = base_tol + (hole_diameter * (scale - 1)) / 2
        This ensures:
        - At scale=1.0: Same tolerance as original (0.0025m)
        - At scale>1.0: Tolerance grows so peg anywhere in hole counts as success
        """
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        # Use per-environment XY tolerance instead of fixed 0.0025
        is_centered = torch.where(
            xy_dist < self.xy_success_tolerance, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )

        # Height threshold (same as parent - not scaled since hole height is unchanged)
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")

        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            import isaacsim.core.utils.torch as torch_utils

            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        # Return training success only
        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """
        Log factory metrics with separate success rates for sim and real.
        """
        super()._log_factory_metrics(rew_dict, curr_successes)

        # Only log at episode boundaries
        if not torch.any(self.reset_buf):
            return

        # Last-frame success rate for all 4 partitions.
        for key in ["train_real", "train_sim", "val_real", "val_sim"]:
            idx = getattr(self, f"idx_{key}")
            num_success = curr_successes[idx].sum().item()
            self.extras[f"success_rate_{key}"] = num_success / max(idx.numel(), 1)

        # Update budget usage based on completed episodes
        # Budget is a bit useless at this moment
        done_sim = self.reset_buf[self.idx_train_sim].sum().item()
        done_real = self.reset_buf[self.idx_train_real].sum().item()
        self.sim_budget_used += done_sim * self.sim_budget
        self.real_budget_used += done_real * self.real_budget
        self.total_budget_used = self.sim_budget_used + self.real_budget_used

        self.extras["budget/sim_used"] = self.sim_budget_used
        self.extras["budget/real_used"] = self.real_budget_used
        self.extras["budget/total_used"] = self.total_budget_used

    def _get_rewards(self):
        """
        Get rewards for all environments.
        """
        return super()._get_rewards()
