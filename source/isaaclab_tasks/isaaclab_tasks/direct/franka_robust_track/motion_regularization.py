"""Finite-difference motion regularizers for robust Cartesian tracking."""

from __future__ import annotations

import torch


def normalized_huber_vector_cost(
    residual: torch.Tensor,
    normalizer: float,
) -> torch.Tensor:
    """Return a component-wise Huber cost averaged across vector dimensions."""

    scale = float(normalizer)
    if scale <= 0.0:
        raise ValueError("Huber normalizer must be positive")
    normalized = residual / scale
    magnitude = normalized.abs()
    cost = torch.where(magnitude <= 1.0, 0.5 * normalized.square(), magnitude - 0.5)
    return cost.mean(dim=-1)


def residual_huber_motion_costs(
    current_linear_acceleration: torch.Tensor,
    current_angular_acceleration: torch.Tensor,
    previous_linear_acceleration: torch.Tensor,
    previous_angular_acceleration: torch.Tensor,
    reference_linear_acceleration: torch.Tensor,
    reference_angular_acceleration: torch.Tensor,
    previous_reference_linear_acceleration: torch.Tensor,
    previous_reference_angular_acceleration: torch.Tensor,
    difference_interval: float,
    jerk_initialized: torch.Tensor,
    linear_acceleration_normalizer: float,
    linear_jerk_normalizer: float,
    angular_acceleration_normalizer: float,
    angular_jerk_normalizer: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return four residual-Huber costs and their untransformed residual norms."""

    interval = float(difference_interval)
    if interval <= 0.0:
        raise ValueError("difference_interval must be positive")

    linear_acceleration_residual = current_linear_acceleration - reference_linear_acceleration
    angular_acceleration_residual = current_angular_acceleration - reference_angular_acceleration
    linear_jerk_residual = (
        (current_linear_acceleration - previous_linear_acceleration)
        - (reference_linear_acceleration - previous_reference_linear_acceleration)
    ) / interval
    angular_jerk_residual = (
        (current_angular_acceleration - previous_angular_acceleration)
        - (reference_angular_acceleration - previous_reference_angular_acceleration)
    ) / interval
    linear_jerk_residual = torch.where(
        jerk_initialized.unsqueeze(-1), linear_jerk_residual, torch.zeros_like(linear_jerk_residual)
    )
    angular_jerk_residual = torch.where(
        jerk_initialized.unsqueeze(-1), angular_jerk_residual, torch.zeros_like(angular_jerk_residual)
    )

    residuals = (
        linear_acceleration_residual,
        linear_jerk_residual,
        angular_acceleration_residual,
        angular_jerk_residual,
    )
    normalizers = (
        linear_acceleration_normalizer,
        linear_jerk_normalizer,
        angular_acceleration_normalizer,
        angular_jerk_normalizer,
    )
    costs = tuple(
        normalized_huber_vector_cost(residual, normalizer)
        for residual, normalizer in zip(residuals, normalizers)
    )
    norms = tuple(torch.linalg.vector_norm(residual, dim=-1) for residual in residuals)
    return (*costs, *norms)


def _raw_vector_acceleration_and_jerk_norms(
    acceleration: torch.Tensor,
    previous_acceleration: torch.Tensor,
    difference_interval: float,
    jerk_initialized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    interval = float(difference_interval)
    if interval <= 0.0:
        raise ValueError("difference_interval must be positive")
    acceleration_norm = torch.linalg.norm(acceleration, dim=-1)
    jerk_norm = torch.linalg.norm(acceleration - previous_acceleration, dim=-1) / interval
    jerk_norm = torch.where(jerk_initialized, jerk_norm, torch.zeros_like(jerk_norm))
    return acceleration_norm, jerk_norm


def raw_linear_acceleration_and_jerk_norms(
    linear_acceleration: torch.Tensor,
    previous_linear_acceleration: torch.Tensor,
    difference_interval: float,
    jerk_initialized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unclipped physical linear EE acceleration and jerk norms."""

    return _raw_vector_acceleration_and_jerk_norms(
        linear_acceleration,
        previous_linear_acceleration,
        difference_interval,
        jerk_initialized,
    )


def raw_angular_acceleration_and_jerk_norms(
    angular_acceleration: torch.Tensor,
    previous_angular_acceleration: torch.Tensor,
    difference_interval: float,
    jerk_initialized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unclipped physical angular EE acceleration and jerk norms."""

    return _raw_vector_acceleration_and_jerk_norms(
        angular_acceleration,
        previous_angular_acceleration,
        difference_interval,
        jerk_initialized,
    )


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
    penalty_mode: str = "hard_clip",
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
    if penalty_mode == "hard_clip":
        if acceleration_clip is not None:
            acceleration_norm = acceleration_norm.clamp(max=float(acceleration_clip))
        if jerk_clip is not None:
            jerk_norm = jerk_norm.clamp(max=float(jerk_clip))
    elif penalty_mode == "log1p":
        if acceleration_clip is None or jerk_clip is None:
            raise ValueError("log1p penalty mode requires positive derivative thresholds")
        acceleration_threshold = float(acceleration_clip)
        jerk_threshold = float(jerk_clip)
        acceleration_norm = acceleration_threshold * torch.log1p(
            acceleration_norm / acceleration_threshold
        )
        jerk_norm = jerk_threshold * torch.log1p(jerk_norm / jerk_threshold)
    else:
        raise ValueError(
            "penalty_mode must be 'hard_clip' or 'log1p', "
            f"got {penalty_mode!r}"
        )
    return acceleration_norm, jerk_norm, linear_acceleration, angular_acceleration
