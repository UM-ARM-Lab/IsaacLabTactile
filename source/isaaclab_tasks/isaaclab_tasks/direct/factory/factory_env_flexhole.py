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

from isaaclab.sensors import TiledCamera

from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskPegInsertFlexHoleCfg


class FactoryFlexHoleEnv(FactoryEnv):
    """Factory environment with mixed large/regular hole sizes.

    This environment creates a mix of environments with different fixed asset (hole) sizes:
    - First `num_large_envs` environments have holes scaled by `large_hole_size`
    - Remaining `num_reg_envs` environments have regular (scale=1.0) holes

    The observations are split into "policy_large" and "policy_reg" for separate processing.
    """

    cfg: FactoryTaskPegInsertFlexHoleCfg

    def __init__(self, cfg: FactoryTaskPegInsertFlexHoleCfg, render_mode: str | None = None, **kwargs):
        # Compute environment counts BEFORE parent init (need num_envs from scene config)
        total_envs = cfg.scene.num_envs
        self.num_large_envs = int(total_envs * cfg.flex_hole.large_env_fraction)
        self.num_reg_envs = total_envs - self.num_large_envs

        # Store scale for later use
        self._large_hole_size = cfg.flex_hole.large_hole_size

        super().__init__(cfg, render_mode, **kwargs)

    def _init_tensors(self):
        """Initialize tensors including hole scale multipliers."""
        super()._init_tensors()

        # Create scale multiplier tensor: [large, large, ..., reg, reg, ...]
        # Shape: (num_envs,)
        large_scales = torch.full((self.num_large_envs,), self._large_hole_size, device=self.device)
        reg_scales = torch.ones(self.num_reg_envs, device=self.device)
        self.hole_scale_multipliers = torch.cat([large_scales, reg_scales])

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

    def _create_multi_scale_fixed_asset(self):
        """Create ArticulationCfg with MultiAssetSpawnerCfg for mixed hole sizes.

        Returns:
            ArticulationCfg with MultiAssetSpawnerCfg spawner containing per-environment
            spawn configurations with appropriate scales.
        """
        base_spawn_cfg = self.cfg_task.fixed_asset.spawn

        # Create spawn configs for each environment
        asset_cfgs = []
        for i in range(self.num_envs):
            scaled_spawn_cfg = deepcopy(base_spawn_cfg)
            if i < self.num_large_envs:
                # Large hole: scale only x,y (not z height)
                scaled_spawn_cfg.scale = (self._large_hole_size, self._large_hole_size, 1.0)
            else:
                # Regular hole: no scaling
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

    def _get_observations(self):
        """Get observations with additional split by hole size.

        Returns:
            dict with keys:
                - "policy": Full policy observations for all envs
                - "critic": Full critic observations for all envs
                - "policy_large": Policy observations for large hole envs only
                - "policy_reg": Policy observations for regular hole envs only
        """
        obs = super()._get_observations()

        # Add split observations by hole size
        policy_obs = obs["policy"]
        obs["policy_large"] = policy_obs[:self.num_large_envs]
        obs["policy_reg"] = policy_obs[self.num_large_envs:]

        return obs
