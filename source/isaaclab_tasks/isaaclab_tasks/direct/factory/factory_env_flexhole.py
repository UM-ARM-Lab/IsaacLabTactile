# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory environment with flexible hole sizes (mixed large/regular environments)."""

from copy import deepcopy
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera, ContactSensor

from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskPegInsertFlexHoleCfg
from . import factory_utils


class FactoryFlexHoleEnv(FactoryEnv):
    """Factory environment with sim/real split and mixed hole sizes.

    Index layout: [sim][real]
    - sim: Scaled hole (easier clearance, for sim-to-real transfer)
    - real: Regular hole (tight clearance, mimics real hardware)

    The environment does not know about "training" vs "validation" - that's
    context-dependent. It only needs to know about "sim" vs "real" hole sizes.
    Both success rates (success_sim, success_real) are logged to wandb.
    """

    cfg: FactoryTaskPegInsertFlexHoleCfg

    def __init__(self, cfg: FactoryTaskPegInsertFlexHoleCfg, render_mode: str | None = None, **kwargs):
        # Parse sim/real counts from flat config
        self.num_sim = cfg.num_sim
        self.num_real = cfg.num_real

        # Validate total matches scene.num_envs
        total_envs = cfg.scene.num_envs
        expected_total = self.num_sim + self.num_real
        if expected_total != total_envs:
            raise ValueError(
                f"Sum of sim/real counts ({expected_total}) must equal scene.num_envs ({total_envs})"
            )

        # Store scale and budget for later use
        self._sim_hole_size = cfg.sim_hole_size
        self.sim_budget = cfg.sim_budget
        self.real_budget = cfg.real_budget

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

        # Budget tracking (episode-based)
        self.sim_budget_used = 0.0
        self.real_budget_used = 0.0
        self.total_budget_used = 0.0

        # Index slices for sim/real
        # Layout: [sim][real]
        self.idx_sim = slice(0, self.num_sim)
        self.idx_real = slice(self.num_sim, None)

        # Create scale multiplier tensor
        # Shape: (num_envs,)
        scales = torch.ones(self.num_envs, device=self.device)
        scales[self.idx_sim] = self._sim_hole_size
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

        # Create MultiAssetSpawnerCfg for fixed_asset with mixed scales
        # fixed_asset_cfg = self._create_multi_scale_fixed_asset()

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

        # Apply per-environment hole scaling via USD API
        self._apply_flex_hole_scales()

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

    def _apply_flex_hole_scales(self):
        """Apply per-environment hole scaling via USD API after clone_environments()."""
        from pxr import Gf
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()

        for i in range(self.num_envs):
            is_sim = i < self.num_sim
            scale = self._sim_hole_size if is_sim else 1.0
            fixed_asset = stage.GetPrimAtPath(f"/World/envs/env_{i}/FixedAsset")
            fixed_asset.GetAttribute("xformOp:scale").Set(Gf.Vec3f(scale, scale, 1.0))

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

        # Return training success only
        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Log factory metrics with separate success rates for sim and real."""
        super()._log_factory_metrics(rew_dict, curr_successes)

        # Only log at episode boundaries
        if not torch.any(self.reset_buf):
            return

        self.extras["success_sim"] = torch.count_nonzero(curr_successes[self.idx_sim]) / max(self.num_sim, 1)
        self.extras["success_real"] = torch.count_nonzero(curr_successes[self.idx_real]) / max(self.num_real, 1)

        # Update budget usage based on completed episodes
        num_done_sim = torch.count_nonzero(self.reset_buf[self.idx_sim]).item()
        num_done_real = torch.count_nonzero(self.reset_buf[self.idx_real]).item()
        self.sim_budget_used += num_done_sim * float(self.sim_budget)
        self.real_budget_used += num_done_real * float(self.real_budget)
        self.total_budget_used = self.sim_budget_used + self.real_budget_used

        self.extras["budget/sim_used"] = self.sim_budget_used
        self.extras["budget/real_used"] = self.real_budget_used
        self.extras["budget/total_used"] = self.total_budget_used

    def _get_rewards(self):
        """Get rewards for all environments."""
        return super()._get_rewards()
