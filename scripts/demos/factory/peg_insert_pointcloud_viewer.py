#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
# noqa: SLF001

"""
Viewer demo for Factory Peg Insertion that visualizes *full* point clouds for:
 - Robot gripper (left + right fingers)
 - Peg (the held asset in the task)

Point clouds are sampled from the USD meshes once (in each body's local frame), then at every
simulation timestep they are transformed using the live rigid-body/root poses so it looks like
perfect (no-occlusion) point clouds coming from a camera.

Usage:
    python scripts/demos/factory/peg_insert_pointcloud_viewer.py --num_envs 1 --env_id 0

Notes:
 - Requires Isaac Sim (pxr) + trimesh to be available in the Isaac Lab python environment.
 - For performance, keep the number of sampled points modest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Factory Peg Insertion viewer with perfect point clouds")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_id", type=int, default=0, help="Which environment index to visualize")
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 = run forever)")
parser.add_argument("--gripper_points", type=int, default=2000, help="Total points for gripper fingers")
parser.add_argument("--peg_points", type=int, default=1000, help="Total points for peg")
parser.add_argument("--point_radius", type=float, default=0.002, help="Marker radius for point visualization")
parser.add_argument(
    "--include_contact_forces",
    default=True,
    action="store_true",
    help="Include fingertip contact forces (left/right, 3D each) in observations and critic states",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app (MUST be before importing other Isaac Lab modules)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of script follows after AppLauncher setup."""

import gymnasium as gym

# Register isaaclab_tasks environments (must import to register gym tasks)
import isaaclab_tasks as _isaaclab_tasks  # noqa: F401

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
from omegaconf import OmegaConf


def _triangulate_face_indices(usd_mesh: Any) -> np.ndarray:
    counts = usd_mesh.GetFaceVertexCountsAttr().Get()
    indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
    faces: list[list[int]] = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)


def _collect_meshes_from_stage(stage: Any, root_path: str):
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []

    meshes = []
    for prim in Usd.PrimRange(root):  # pyright: ignore[reportAttributeAccessIssue]
        if prim.GetTypeName() != "Mesh":
            continue
        usd_mesh = UsdGeom.Mesh(prim)  # pyright: ignore[reportAttributeAccessIssue]
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
        if points.size == 0:
            continue

        face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        if face_counts:
            faces = _triangulate_face_indices(usd_mesh)
        else:
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            faces = (
                np.asarray(indices, dtype=np.int64).reshape(-1, 3) if len(indices) else np.zeros((0, 3), dtype=np.int64)
            )

        if len(faces) > 0:
            import trimesh

            meshes.append(trimesh.Trimesh(vertices=points, faces=faces, process=False))
    return meshes


def _load_meshes_from_usd(usd_path: Path, prim_path: str) -> list:
    if not usd_path.exists():
        return []
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))  # pyright: ignore[reportAttributeAccessIssue]
    return _collect_meshes_from_stage(stage, prim_path)


def _sample_meshes_to_points(meshes: list, num_points: int) -> np.ndarray:
    if not meshes:
        return np.zeros((0, 3), dtype=np.float64)
    total_area = float(sum(m.area for m in meshes))
    if total_area <= 0.0:
        return np.zeros((0, 3), dtype=np.float64)

    from trimesh.sample import sample_surface

    all_points = []
    for mesh in meshes:
        n = max(1, int(num_points * float(mesh.area) / total_area))
        points = sample_surface(mesh, n, face_weight=mesh.area_faces)[0]
        all_points.append(np.asarray(points, dtype=np.float64))
    return np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float64)


def _resolve_factory_assets_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    return repo_root / "assets" / "Factory_new"


def _load_local_pointclouds_from_assets(gripper_points: int, peg_points: int):
    asset_dir = _resolve_factory_assets_dir()
    robot_usd = asset_dir / "franka_gelsight_r15_assembled.usd"
    peg_usd = asset_dir / "factory_peg_8mm.usd"

    left_meshes = _load_meshes_from_usd(robot_usd, "/panda/panda_leftfinger/elastomer")
    right_meshes = _load_meshes_from_usd(robot_usd, "/panda/panda_rightfinger/elastomer")
    if not left_meshes or not right_meshes:
        meshes = _load_meshes_from_usd(robot_usd, "/panda")
        half = max(1, gripper_points // 2)
        left_pts = _sample_meshes_to_points(meshes, half)
        right_pts = _sample_meshes_to_points(meshes, gripper_points - half)
    else:
        half = max(1, gripper_points // 2)
        left_pts = _sample_meshes_to_points(left_meshes, half)
        right_pts = _sample_meshes_to_points(right_meshes, gripper_points - half)

    if peg_usd.exists():
        from pxr import Usd

        stage = Usd.Stage.Open(str(peg_usd))  # pyright: ignore[reportAttributeAccessIssue]
        default_prim = stage.GetDefaultPrim()
        root = default_prim.GetPath().pathString if default_prim else "/"
        peg_meshes = _collect_meshes_from_stage(stage, root)
    else:
        peg_meshes = []
    peg_pts = _sample_meshes_to_points(peg_meshes, peg_points)

    return left_pts, right_pts, peg_pts


def main():
    if args_cli.env_id < 0 or args_cli.env_id >= args_cli.num_envs:
        raise ValueError(f"--env_id must be in [0, {args_cli.num_envs - 1}]")

    # Create a config with tactile disabled (viewer-only)
    env_cfg = FactoryTaskPegInsertCfg(params=OmegaConf.create({"env": {}}))
    env_cfg.enable_tactile_sensor = False
    env_cfg.read_tactile_sensor = False
    env_cfg.enable_obs_camera = False
    env_cfg.use_compliant_gripper = True
    env_cfg.use_gelsight_finger = True
    env_cfg.scene.num_envs = args_cli.num_envs
    # Optionally include fingertip contact forces in observations and critic states
    env_cfg.include_contact_forces = bool(args_cli.include_contact_forces)

    if getattr(args_cli, "device", None) == "cpu":
        env_cfg.sim.device = "cpu"
    elif getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    print("[INFO] Creating Factory Peg Insert environment...")
    env: FactoryEnv = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=env_cfg)
    _ = env.reset()
    print("[INFO] Environment reset.")

    sim_device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cpu"

    print("[INFO] Loading and sampling local-frame point clouds from USD assets...")
    left_pts_np, right_pts_np, peg_pts_np = _load_local_pointclouds_from_assets(
        gripper_points=args_cli.gripper_points,
        peg_points=args_cli.peg_points,
    )
    left_pts_l = torch.tensor(left_pts_np, dtype=torch.float32, device=sim_device)
    right_pts_l = torch.tensor(right_pts_np, dtype=torch.float32, device=sim_device)
    peg_pts_l = torch.tensor(peg_pts_np, dtype=torch.float32, device=sim_device)

    left_body_idx = env.unwrapped._robot.body_names.index("elastomer")
    right_body_idx = env.unwrapped._robot.body_names.index("elastomer_0")

    has_gui = not bool(getattr(args_cli, "headless", False))

    if not has_gui:
        print("[WARN] No GUI available; point cloud visualization will be disabled.")
        pc_gripper = None
        pc_peg = None
    else:
        cfg_gripper = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/Factory/GripperPointCloud")
        cfg_gripper.markers["hit"].radius = args_cli.point_radius
        cfg_gripper.markers["hit"].visual_material.diffuse_color = (0.8, 0.2, 0.2)
        pc_gripper = VisualizationMarkers(cfg_gripper)

        cfg_peg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/Factory/PegPointCloud")
        cfg_peg.markers["hit"].radius = args_cli.point_radius
        cfg_peg.markers["hit"].visual_material.diffuse_color = (0.2, 0.8, 0.2)
        pc_peg = VisualizationMarkers(cfg_peg)

    action_dim = env.unwrapped.action_size if hasattr(env.unwrapped, "action_size") else 6
    zero_actions = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=sim_device)
    random_actions = torch.rand((args_cli.num_envs, action_dim), dtype=torch.float32, device=sim_device) * 1.0

    print("[INFO] Starting viewer loop. Close viewer or Ctrl+C to exit.")
    steps = 0
    while simulation_app.is_running():
        _ = env.step(random_actions)

        if pc_gripper is not None and pc_peg is not None:
            e = args_cli.env_id

            left_pos_w = env.unwrapped._robot.data.body_pos_w[e, left_body_idx]
            left_quat_w = env.unwrapped._robot.data.body_quat_w[e, left_body_idx]
            right_pos_w = env.unwrapped._robot.data.body_pos_w[e, right_body_idx]
            right_quat_w = env.unwrapped._robot.data.body_quat_w[e, right_body_idx]

            peg_pos_w = env.unwrapped._held_asset.data.root_pos_w[e]
            peg_quat_w = env.unwrapped._held_asset.data.root_quat_w[e]

            left_pts_w = math_utils.quat_apply(left_quat_w.unsqueeze(0), left_pts_l) + left_pos_w.unsqueeze(0)
            right_pts_w = math_utils.quat_apply(right_quat_w.unsqueeze(0), right_pts_l) + right_pos_w.unsqueeze(0)
            gripper_pts_w = torch.cat([left_pts_w, right_pts_w], dim=0)

            peg_pts_w = math_utils.quat_apply(peg_quat_w.unsqueeze(0), peg_pts_l) + peg_pos_w.unsqueeze(0)

            if gripper_pts_w.shape[0] > 0:
                pc_gripper.visualize(translations=gripper_pts_w)
            if peg_pts_w.shape[0] > 0:
                pc_peg.visualize(translations=peg_pts_w)

        steps += 1
        if args_cli.steps > 0 and steps >= args_cli.steps:
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

