# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory environment with flexible hole sizes (mixed large/regular environments)."""

from copy import deepcopy
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera, ContactSensor

from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskPegInsertFlexHoleCfg
from . import factory_utils


class FactoryFlexHoleEnv(FactoryEnv):
    """Factory environment with train/val split and mixed hole sizes.

    Index layout: [train_large][train_reg][val_large][val_reg]
    - train_large: sim-train (scaled hole, easier)
    - train_reg: real-train (regular hole, harder)
    - val_large: sim-val (scaled hole, for monitoring)
    - val_reg: real-val (regular hole, for monitoring)

    Training uses only train envs (obs/rewards filtered). Val envs run for monitoring only.
    All 4 success rates are always logged to wandb.
    """

    cfg: FactoryTaskPegInsertFlexHoleCfg

    def __init__(self, cfg: FactoryTaskPegInsertFlexHoleCfg, render_mode: str | None = None, **kwargs):
        # Parse train/val counts from config
        flex = cfg.flex_hole
        self.num_train_large = flex.num_train_large
        self.num_train_reg = flex.num_train_reg
        self.num_val_large = flex.num_val_large
        self.num_val_reg = flex.num_val_reg

        # Computed counts
        self.num_train = self.num_train_large + self.num_train_reg
        self.num_val = self.num_val_large + self.num_val_reg
        self.num_large_envs = self.num_train_large + self.num_val_large
        self.num_reg_envs = self.num_train_reg + self.num_val_reg

        # Validate total matches scene.num_envs
        total_envs = cfg.scene.num_envs
        expected_total = self.num_train + self.num_val
        if expected_total != total_envs:
            raise ValueError(
                f"Sum of flex_hole counts ({expected_total}) must equal scene.num_envs ({total_envs})"
            )

        # Store scale for later use
        self._large_hole_size = flex.large_hole_size

        # Pre-adjust observation_space for 6D quaternion representation
        # Each quaternion field (4D) becomes 6D, adding 2 dimensions per field
        num_quat_fields_obs = sum(1 for obs in cfg.obs_order if obs.endswith("_quat"))
        num_quat_fields_state = sum(1 for s in cfg.state_order if s.endswith("_quat"))
        self._obs_quat_adjustment = num_quat_fields_obs * 2
        self._state_quat_adjustment = num_quat_fields_state * 2

        super().__init__(cfg, render_mode, **kwargs)

        # Adjust the computed observation_space and state_space for 6D quaternions
        self.cfg.observation_space += self._obs_quat_adjustment
        self.cfg.state_space += self._state_quat_adjustment

        # Update the gym observation space Dict to match new dimensions
        from gymnasium import spaces
        import numpy as np
        new_obs_dim = self.cfg.observation_space
        new_state_dim = self.cfg.state_space
        self.single_observation_space["policy"] = spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,))
        self.single_observation_space["critic"] = spaces.Box(low=-np.inf, high=np.inf, shape=(new_state_dim,))

    def _init_tensors(self):
        """Initialize tensors including hole scale multipliers and tolerances."""
        super()._init_tensors()

        # Index slices for each category
        # Layout: [train_large][train_reg][val_large][val_reg]
        self.idx_train_large = slice(0, self.num_train_large)
        self.idx_train_reg = slice(self.num_train_large, self.num_train)
        self.idx_val_large = slice(self.num_train, self.num_train + self.num_val_large)
        self.idx_val_reg = slice(self.num_train + self.num_val_large, None)
        self.idx_train = slice(0, self.num_train)
        self.idx_val = slice(self.num_train, None)

        # Create scale multiplier tensor for new layout
        # Shape: (num_envs,)
        scales = torch.ones(self.num_envs, device=self.device)
        scales[self.idx_train_large] = self._large_hole_size
        scales[self.idx_val_large] = self._large_hole_size
        self.hole_scale_multipliers = scales

        # Compute per-environment XY success tolerance using affine formula:
        # xy_tolerance = base_tolerance + (hole_diameter * (scale - 1)) / 2
        # At scale=1.0: tolerance = 0.0025 (unchanged from original)
        # At scale>1.0: tolerance grows to accommodate larger hole
        base_xy_tolerance = 0.0025  # Original hardcoded tolerance
        hole_diameter = self.cfg_task.fixed_asset_cfg.diameter
        self.xy_success_tolerance = base_xy_tolerance + (hole_diameter * (self.hole_scale_multipliers - 1.0)) / 2

    def _setup_scene(self):
        """Initialize simulation scene with mixed hole sizes using MultiAssetSpawnerCfg."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Spawn table
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg,
            translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        # Clone environments FIRST (required for MultiAssetSpawnerCfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        # Create MultiAssetSpawnerCfg for fixed_asset with mixed scales
        fixed_asset_cfg = self._create_multi_scale_fixed_asset()

        # Spawn assets
        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(fixed_asset_cfg)
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

    def _create_multi_scale_fixed_asset(self):
        """Create ArticulationCfg with MultiAssetSpawnerCfg for mixed hole sizes.

        Layout: [train_large][train_reg][val_large][val_reg]
        Large holes at train_large and val_large indices.

        Returns:
            ArticulationCfg with MultiAssetSpawnerCfg spawner containing per-environment
            spawn configurations with appropriate scales.
        """
        base_spawn_cfg = self.cfg_task.fixed_asset.spawn

        # Create spawn configs for each environment
        asset_cfgs = []
        for i in range(self.num_envs):
            scaled_spawn_cfg = deepcopy(base_spawn_cfg)
            # Large hole for train_large or val_large indices
            is_train_large = i < self.num_train_large
            is_val_large = self.num_train <= i < self.num_train + self.num_val_large
            if is_train_large or is_val_large:
                scaled_spawn_cfg.scale = (self._large_hole_size, self._large_hole_size, 1.0)
            else:
                scaled_spawn_cfg.scale = (1.0, 1.0, 1.0)
            asset_cfgs.append(scaled_spawn_cfg)

        # Create MultiAssetSpawnerCfg
        multi_asset_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=asset_cfgs,
            random_choice=False,  # Deterministic: env i uses asset_cfgs[i]
            activate_contact_sensors=True,
        )

        # Create new ArticulationCfg with multi-asset spawner
        fixed_asset_cfg = deepcopy(self.cfg_task.fixed_asset)
        fixed_asset_cfg.spawn = multi_asset_spawn_cfg
        return fixed_asset_cfg

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
        v1 = 1 - 2*y*y - 2*z*z
        v2 = 2*x*y + 2*z*w
        v3 = 2*x*z - 2*y*w

        # Second column of rotation matrix
        v4 = 2*x*y - 2*z*w
        v5 = 1 - 2*x*x - 2*z*z
        v6 = 2*y*z + 2*x*w

        return torch.stack([v1, v2, v3, v4, v5, v6], dim=-1)


    def _get_observations(self):
        """Get observations with 6D quaternion representation.

        All environments (train and val) contribute to training.
        Success rates are logged separately for monitoring.

        Returns:
            dict with keys:
                - "policy": Policy observations for all envs
                - "critic": Critic observations for all envs
        """
        obs_dict, state_dict, collect_dict = super()._get_factory_obs_state_dict()
        # Replace keys with "quat" with 6D representation
        for d in [obs_dict, state_dict, collect_dict]:
            for key in d.keys():
                if not key.endswith("_quat"):
                    continue
                quat_tensor = d[key]
                d[key] = self.quat_to_6d(quat_tensor)

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        # Store collection observations for data collection
        self.collect_obs = torch.cat([collect_dict[key] for key in collect_dict.keys()], dim=-1)

        # Return observations for ALL environments (no filtering)
        # Val envs contribute to training but have separate success rate logging
        obs = {}
        obs["policy"] = obs_tensors
        obs["critic"] = state_tensors

        return obs

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
            xy_dist < self.xy_success_tolerance,
            torch.ones_like(curr_successes),
            torch.zeros_like(curr_successes)
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

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Log factory metrics with separate success rates for all 4 categories."""
        super()._log_factory_metrics(rew_dict, curr_successes)

        # Log 4 categories only at episode boundaries (matching parent's behavior for smooth curves)
        if torch.any(self.reset_buf):
            self.extras["successes_train_large"] = torch.count_nonzero(curr_successes[self.idx_train_large]) / max(self.num_train_large, 1)
            self.extras["successes_train_reg"] = torch.count_nonzero(curr_successes[self.idx_train_reg]) / max(self.num_train_reg, 1)
            self.extras["successes_val_large"] = torch.count_nonzero(curr_successes[self.idx_val_large]) / max(self.num_val_large, 1)
            self.extras["successes_val_reg"] = torch.count_nonzero(curr_successes[self.idx_val_reg]) / max(self.num_val_reg, 1)

    def _get_rewards(self):
        """Get rewards for all environments.

        All envs (train and val) contribute to training.
        Success rates are logged separately for monitoring.
        """
        return super()._get_rewards()
