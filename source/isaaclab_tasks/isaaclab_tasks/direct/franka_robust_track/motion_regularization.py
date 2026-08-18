"""Finite-difference motion regularizers for robust Cartesian tracking."""

from __future__ import annotations

import torch


def ee_acceleration_and_jerk_norms(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    previous_linear_velocity: torch.Tensor,
    previous_angular_velocity: torch.Tensor,
    previous_linear_acceleration: torch.Tensor,
    previous_angular_acceleration: torch.Tensor,
    difference_interval: float,
    jerk_initialized: torch.Tensor,
    acceleration_clip: float | None = None,
    jerk_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return EE acceleration/jerk norms and current accelerations.

    ``difference_interval`` is the policy timestep for physical derivatives.
    Passing the inverse native-rate normalization reproduces the historical
    dimensionless finite differences when migrating an older configuration.
    ``jerk_initialized`` suppresses the undefined first jerk sample following
    an environment reset.
    """

    interval = float(difference_interval)
    if interval <= 0.0:
        raise ValueError("difference_interval must be positive")
    linear_acceleration = (linear_velocity - previous_linear_velocity) / interval
    angular_acceleration = (angular_velocity - previous_angular_velocity) / interval
    acceleration_norm = torch.linalg.norm(linear_acceleration, dim=-1) + 0.1 * torch.linalg.norm(
        angular_acceleration, dim=-1
    )
    jerk_norm = (
        torch.linalg.norm(linear_acceleration - previous_linear_acceleration, dim=-1)
        + 0.1
        * torch.linalg.norm(angular_acceleration - previous_angular_acceleration, dim=-1)
    ) / interval
    jerk_norm = torch.where(jerk_initialized, jerk_norm, torch.zeros_like(jerk_norm))
    if acceleration_clip is not None:
        acceleration_norm = acceleration_norm.clamp(max=float(acceleration_clip))
    if jerk_clip is not None:
        jerk_norm = jerk_norm.clamp(max=float(jerk_clip))
    return acceleration_norm, jerk_norm, linear_acceleration, angular_acceleration
