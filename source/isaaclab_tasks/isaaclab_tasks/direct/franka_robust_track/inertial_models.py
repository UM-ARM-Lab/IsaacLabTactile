"""Load rigid-body inertial parameters from the controller's URDF model."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def load_urdf_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Return ``link -> (mass, CoM position, inertia in the link frame)``."""
    result = {}
    for link in ET.parse(path).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        xyz = np.fromstring(
            "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0"), sep=" "
        )
        rpy = np.fromstring(
            "0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0"), sep=" "
        )
        values = inertial.find("inertia").attrib
        inertia = np.array(
            [
                [float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                [float(values["ixz"]), float(values["iyz"]), float(values["izz"])],
            ],
            dtype=np.float64,
        )
        rotation = _rpy_matrix(rpy)
        result[link.attrib["name"]] = (
            float(inertial.find("mass").attrib["value"]),
            xyz,
            rotation @ inertia @ rotation.T,
        )
    return result


def map_urdf_inertials(
    body_names: list[str], path: str, aliases: dict[str, str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a URDF model onto simulation bodies; unlisted bodies remain zero."""
    inertials = load_urdf_inertials(path)
    aliases = aliases or {}
    masses = np.zeros(len(body_names), dtype=np.float64)
    coms = np.zeros((len(body_names), 3), dtype=np.float64)
    tensors = np.zeros((len(body_names), 3, 3), dtype=np.float64)
    present = np.zeros(len(body_names), dtype=bool)
    for index, body_name in enumerate(body_names):
        urdf_name = aliases.get(body_name, body_name)
        if urdf_name not in inertials:
            continue
        masses[index], coms[index], tensors[index] = inertials[urdf_name]
        present[index] = True
    return masses, coms, tensors, present
