"""Pure dual-clock indexing helpers for robust trajectory tracking."""

from __future__ import annotations

import torch


def axis_aligned_containment_mask(
    position: torch.Tensor, box: torch.Tensor
) -> torch.Tensor:
    """Return whether each position lies inside an XYZ ``(3, 2)`` box."""
    if position.shape[-1] != 3 or box.shape != (3, 2):
        raise ValueError("position must end in 3 and box must have shape (3, 2)")
    return ((position >= box[:, 0]) & (position <= box[:, 1])).all(dim=-1)


def policy_step_to_reference_index(
    policy_step: torch.Tensor, policy_steps_per_reference: int
) -> torch.Tensor:
    """Map policy counters to native reference indices with zero-order hold."""
    if policy_steps_per_reference < 1:
        raise ValueError("policy_steps_per_reference must be positive")
    return torch.div(
        policy_step, policy_steps_per_reference, rounding_mode="floor"
    )


def is_reference_boundary(
    policy_step: torch.Tensor, policy_steps_per_reference: int
) -> torch.Tensor:
    """Return whether each policy counter lies on a native reference sample."""
    if policy_steps_per_reference < 1:
        raise ValueError("policy_steps_per_reference must be positive")
    return torch.remainder(policy_step, policy_steps_per_reference) == 0


def history_seed_offsets(
    history_length: int, policy_steps_per_reference: int
) -> torch.Tensor:
    """Return pre-roll dataset-history offsets in native-sample units.

    The environment rolls once and appends the current state after reset. Thus,
    for a 30 Hz policy with a three-frame history this returns
    ``[-1.5, -1.0, -0.5]``; the first observation then contains
    ``[-1.0, -0.5, 0.0]`` and preserves the original 15 Hz history span.
    """
    if history_length < 1:
        raise ValueError("history_length must be positive")
    if policy_steps_per_reference < 1:
        raise ValueError("policy_steps_per_reference must be positive")
    return torch.arange(-history_length, 0, dtype=torch.float32) / float(
        policy_steps_per_reference
    )


def reference_coordinates(
    policy_step: torch.Tensor,
    *,
    policy_decimation: int,
    reference_decimation: int,
    completed_physics_substeps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native endpoints and phase at a policy/physics clock instant."""
    if policy_decimation < 1:
        raise ValueError("policy_decimation must be positive")
    if reference_decimation < policy_decimation:
        raise ValueError("reference_decimation must be at least policy_decimation")
    if reference_decimation % policy_decimation:
        raise ValueError(
            "reference_decimation must be divisible by policy_decimation"
        )
    if not 0 <= completed_physics_substeps <= policy_decimation:
        raise ValueError(
            "completed_physics_substeps must be in [0, policy_decimation]"
        )
    total_physics_steps = (
        policy_step * policy_decimation + completed_physics_substeps
    )
    i0 = torch.div(
        total_physics_steps, reference_decimation, rounding_mode="floor"
    )
    remainder = torch.remainder(total_physics_steps, reference_decimation)
    phase = remainder.float() / float(reference_decimation)
    return i0, i0 + 1, phase


def virtual_contact_reference_coordinates(
    policy_step: torch.Tensor,
    *,
    policy_decimation: int,
    reference_decimation: int,
    completed_physics_substeps: int = 0,
    interpolate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return virtual-contact coordinates for interpolation or native-rate hold."""
    i0, i1, phase = reference_coordinates(
        policy_step,
        policy_decimation=policy_decimation,
        reference_decimation=reference_decimation,
        completed_physics_substeps=completed_physics_substeps,
    )
    if not interpolate:
        i1 = i0
        phase = torch.zeros_like(phase)
    return i0, i1, phase
