#!/usr/bin/env python3
"""
Temporary script to load the fingertip (gripper) meshes and the peg mesh from USD,
sample surface points to form point clouds (with correct link poses applied),
and visualize them.

Run from IsaacLabTactile repo root (or ensure assets/Factory_new paths exist):
    python scripts/demos/factory/visualize_gripper_peg_meshes.py

Requires: pxr (usd-core or Isaac Sim), numpy, trimesh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom
from trimesh.sample import sample_surface

# Resolve asset dir: script is at scripts/demos/factory/ -> repo root is parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
ASSET_DIR = _REPO_ROOT / "assets" / "Factory_new"

# Default USD paths (can be overridden)
ROBOT_USD = ASSET_DIR / "franka_mimic.usd"
PEG_USD = ASSET_DIR / "factory_peg_8mm.usd"
# If tactile robot is present, use it for more detailed finger meshes
ROBOT_TACTILE_USD = ASSET_DIR / "franka_gelsight_r15_assembled.usd"

# Point cloud sampling: total number of points to sample (distributed by mesh surface area)
NUM_SAMPLE_POINTS_GRIPPER = 2000
NUM_SAMPLE_POINTS_PEG = 1000


def _triangulate_face_indices(usd_mesh: UsdGeom.Mesh) -> np.ndarray:
    """Convert USD mesh face vertex indices to (N, 3) triangle indices."""
    counts = usd_mesh.GetFaceVertexCountsAttr().Get()
    indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)


def collect_meshes_from_stage(stage: Usd.Stage, root_path: str):
    """Recursively collect all Mesh prims under root_path and return a list of trimesh.Trimesh."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []

    meshes = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)

        if points.size == 0:
            continue

        face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        if face_counts:
            faces = _triangulate_face_indices(usd_mesh)
        else:
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3) if len(indices) else np.zeros((0, 3), dtype=np.int64)

        if len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            meshes.append(mesh)
    return meshes


def load_gripper_meshes(usd_path: Path, root_path: str = "/panda") -> list:
    """Load all meshes under the gripper (finger) part of the robot USD."""
    if not usd_path.exists():
        return []
    stage = Usd.Stage.Open(str(usd_path))
    # Collect from both fingers if they exist; otherwise from whole robot
    all_meshes = []
    for sub in ("/panda/panda_leftfinger/elastomer", "/panda/panda_rightfinger/elastomer"):
        prim = stage.GetPrimAtPath(sub)
        if prim and prim.IsValid():
            all_meshes.extend(collect_meshes_from_stage(stage, sub))
    if not all_meshes:
        all_meshes = collect_meshes_from_stage(stage, root_path)
    return all_meshes


def load_peg_meshes(usd_path: Path) -> list:
    """Load all meshes from the peg USD (typically a single mesh). Fallback to OBJ if USD missing."""
    if usd_path.exists():
        stage = Usd.Stage.Open(str(usd_path))
        default_prim = stage.GetDefaultPrim()
        root = default_prim.GetPath().pathString if default_prim else "/"
        return collect_meshes_from_stage(stage, root)
    # Fallback: load from OBJ if USD not present (e.g. factory_peg_8mm.obj in objs/)
    obj_path = usd_path.parent / "objs" / (usd_path.stem + ".obj")
    if obj_path.exists():
        mesh = trimesh.load(str(obj_path))
        if isinstance(mesh, trimesh.Trimesh):
            return [mesh]
        if isinstance(mesh, trimesh.Scene):
            return [m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)]
    return []


def sample_meshes_to_point_cloud(meshes: list, num_points: int) -> np.ndarray:
    """
    Sample points on the surface of meshes with area-weighted distribution.
    Meshes are assumed to have vertices already in world space (link poses applied).
    Returns (N, 3) array of points in world frame.
    """
    if not meshes:
        return np.zeros((0, 3), dtype=np.float64)
    total_area = sum(m.area for m in meshes)
    if total_area <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    all_points = []
    for mesh in meshes:
        n = max(1, int(num_points * mesh.area / total_area))
        face_weights = mesh.area_faces
        points, _ = sample_surface(mesh, n, face_weight=face_weights)
        all_points.append(np.asarray(points, dtype=np.float64))
    return np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float64)


def get_gripper_and_peg_point_clouds(
    robot_usd: Path | None = None,
    peg_usd: Path | None = None,
    num_gripper_points: int = NUM_SAMPLE_POINTS_GRIPPER,
    num_peg_points: int = NUM_SAMPLE_POINTS_PEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load meshes, sample surface points with link poses applied, return point clouds.
    Returns (gripper_points, peg_points) each (N, 3) in world frame.
    """
    robot_usd = robot_usd or (ROBOT_TACTILE_USD if ROBOT_TACTILE_USD.exists() else ROBOT_USD)
    peg_usd = peg_usd or PEG_USD
    gripper_meshes = load_gripper_meshes(robot_usd)
    peg_meshes = load_peg_meshes(peg_usd)
    gripper_points = sample_meshes_to_point_cloud(gripper_meshes, num_gripper_points)
    peg_points = sample_meshes_to_point_cloud(peg_meshes, num_peg_points)
    return gripper_points, peg_points


def main():
    # Prefer tactile robot if present (has detailed finger meshes)
    robot_usd = ROBOT_TACTILE_USD if ROBOT_TACTILE_USD.exists() else ROBOT_USD
    print(f"Robot USD: {robot_usd} (exists: {robot_usd.exists()})")
    print(f"Peg USD:  {PEG_USD} (exists: {PEG_USD.exists()})")

    gripper_meshes = load_gripper_meshes(robot_usd)
    peg_meshes = load_peg_meshes(PEG_USD)

    if not gripper_meshes and not peg_meshes:
        print("No meshes found. Check that the USD files exist and contain Mesh prims.")
        return

    # Sample point clouds from meshes (vertices are already in world frame from link poses)
    gripper_points = sample_meshes_to_point_cloud(gripper_meshes, NUM_SAMPLE_POINTS_GRIPPER)
    peg_points = sample_meshes_to_point_cloud(peg_meshes, NUM_SAMPLE_POINTS_PEG)

    # Offset peg so it doesn't overlap the gripper in the viewer
    peg_offset = np.array([0.15, 0.0, 0.0])
    peg_points = peg_points + peg_offset

    # Build scene with point clouds (distinct colors)
    scene_geoms = []
    if len(gripper_points) > 0:
        pc_gripper = trimesh.PointCloud(gripper_points, colors=[180, 80, 80, 255])
        scene_geoms.append(pc_gripper)
    if len(peg_points) > 0:
        pc_peg = trimesh.PointCloud(peg_points, colors=[80, 180, 80, 255])
        scene_geoms.append(pc_peg)

    scene = trimesh.Scene(scene_geoms)
    print(
        f"Loaded {len(gripper_meshes)} gripper mesh(es) -> {len(gripper_points)} points, "
        f"{len(peg_meshes)} peg mesh(es) -> {len(peg_points)} points. Opening viewer..."
    )
    scene.show()


if __name__ == "__main__":
    main()
