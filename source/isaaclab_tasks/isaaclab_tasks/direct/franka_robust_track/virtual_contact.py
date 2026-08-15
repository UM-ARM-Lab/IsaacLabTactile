"""Small tensor utilities for RobustTrack's geometry-free contact model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit ``wxyz`` quaternions with matching leading shapes."""
    if quat.shape[:-1] != vector.shape[:-1] or quat.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError(
            "quat/vector must have matching leading shapes and trailing dimensions 4/3; "
            f"got {tuple(quat.shape)} and {tuple(vector.shape)}"
        )
    xyz = quat[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quat[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def sensor_reaction_to_world_external(
    sensor_reaction: torch.Tensor,
    sensor_quat_world: torch.Tensor,
) -> torch.Tensor:
    """Convert a sensor-local incoming-joint reaction to a world external vector.

    PhysX reports ``get_link_incoming_joint_force`` in the child-joint frame. For
    the Forge force-sensor joint that frame is aligned with the sensor link. The
    external wrench that generated that joint reaction has the opposite sign.
    The input may contain either one three-vector or a six-axis force/torque
    wrench; force and torque are rotated independently.
    """
    if sensor_reaction.shape[-1] == 3:
        return -_quat_apply_wxyz(sensor_quat_world, sensor_reaction)
    if sensor_reaction.shape[-1] != 6:
        raise ValueError(
            "sensor_reaction must have trailing dimension 3 or 6, "
            f"got {tuple(sensor_reaction.shape)}"
        )
    return -torch.cat(
        (
            _quat_apply_wxyz(sensor_quat_world, sensor_reaction[..., :3]),
            _quat_apply_wxyz(sensor_quat_world, sensor_reaction[..., 3:]),
        ),
        dim=-1,
    )


def world_external_to_sensor_reaction(
    world_external: torch.Tensor,
    sensor_quat_world: torch.Tensor,
) -> torch.Tensor:
    """Convert a world external force or wrench to its sensor-local reaction."""
    sensor_quat_inverse = torch.cat(
        (sensor_quat_world[..., :1], -sensor_quat_world[..., 1:]), dim=-1
    )
    if world_external.shape[-1] == 3:
        return -_quat_apply_wxyz(sensor_quat_inverse, world_external)
    if world_external.shape[-1] != 6:
        raise ValueError(
            "world_external must have trailing dimension 3 or 6, "
            f"got {tuple(world_external.shape)}"
        )
    return -torch.cat(
        (
            _quat_apply_wxyz(sensor_quat_inverse, world_external[..., :3]),
            _quat_apply_wxyz(sensor_quat_inverse, world_external[..., 3:]),
        ),
        dim=-1,
    )


def contact_coupled_torque(
    target_force_world: torch.Tensor,
    target_torque_world: torch.Tensor,
    actual_force_world: torch.Tensor,
    in_contact: torch.Tensor,
    max_multiplier: float,
    torque_cap: float | None,
) -> torch.Tensor:
    """Scale a demonstrated torque with the realized virtual-contact force.

    At the demonstrated/reference pose the realized and target force magnitudes
    agree, so the demonstrated torque is reproduced. Moving out of contact makes
    both force and torque zero; deeper penetration strengthens both. A target with
    negligible force falls back to a binary contact gate so a torsional target is
    still representable.
    """
    for name, value in (
        ("target_force_world", target_force_world),
        ("target_torque_world", target_torque_world),
        ("actual_force_world", actual_force_world),
    ):
        if value.shape[-1] != 3:
            raise ValueError(f"{name} must have trailing dimension 3, got {tuple(value.shape)}")
    if target_force_world.shape != target_torque_world.shape or target_force_world.shape != actual_force_world.shape:
        raise ValueError("target force, target torque, and actual force must have matching shapes")
    if in_contact.shape != target_force_world.shape[:-1]:
        raise ValueError(
            "in_contact must match the wrench leading shape, "
            f"got {tuple(in_contact.shape)} and {tuple(target_force_world.shape[:-1])}"
        )

    target_force_mag = torch.linalg.norm(target_force_world, dim=-1, keepdim=True)
    actual_force_mag = torch.linalg.norm(actual_force_world, dim=-1, keepdim=True)
    force_ratio = actual_force_mag / target_force_mag.clamp(min=1.0e-6)
    ratio = torch.where(
        target_force_mag > 1.0e-6,
        force_ratio,
        in_contact.unsqueeze(-1).to(target_force_world.dtype),
    )
    ratio = torch.where(
        in_contact.unsqueeze(-1),
        ratio,
        torch.zeros_like(ratio),
    )
    if float(max_multiplier) > 0.0:
        ratio = ratio.clamp(max=float(max_multiplier))
    torque = target_torque_world * ratio

    if torque_cap is not None and float(torque_cap) > 0.0:
        magnitude = torch.linalg.norm(torque, dim=-1, keepdim=True)
        torque = torque * (float(torque_cap) / magnitude.clamp(min=1.0e-6)).clamp(max=1.0)
    return torque


def invert_ema_sequence(smoothed: torch.Tensor, alpha: float) -> torch.Tensor:
    """Recover the unsmoothed sequence for an EMA initialized at zero.

    The forward filter is ``y[t] = alpha*x[t] + (1-alpha)*y[t-1]`` with
    ``y[-1] = 0``.  The inverse is exact apart from floating-point roundoff and
    is applied independently to every trailing component (including torque).
    """
    if smoothed.ndim < 2:
        raise ValueError(
            f"smoothed must have a batch and time dimension, got {tuple(smoothed.shape)}"
        )
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    previous = torch.cat((torch.zeros_like(smoothed[:, :1]), smoothed[:, :-1]), dim=1)
    return (smoothed - (1.0 - alpha) * previous) / alpha


def power_law_coefficient_from_reference(
    reference_force: float,
    reference_penetration: float,
    exponent: float,
) -> float:
    """Return ``k_p = f_ref / delta_ref**p`` for a power-law contact."""
    reference_force = float(reference_force)
    reference_penetration = float(reference_penetration)
    exponent = float(exponent)
    if reference_force <= 0.0:
        raise ValueError("reference_force must be positive")
    if reference_penetration <= 0.0:
        raise ValueError("reference_penetration must be positive")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    return reference_force / reference_penetration**exponent


def smooth_force_directions(
    force: torch.Tensor,
    window: int,
    valid_force_threshold: float,
) -> torch.Tensor:
    """Return temporally smooth unit force directions for complete trajectories.

    Args:
        force: ``(batch, time, 3)`` world-frame force trajectories.
        window: Odd centered moving-average window. ``1`` disables smoothing.
        valid_force_threshold: A smoothed force below this magnitude does not
            provide a trustworthy direction. Such samples hold the most recent
            valid direction; samples before first contact use the first valid
            direction in the trajectory.

    The complete reference trajectory is known at reset, so a centered filter
    avoids the phase lag of a causal EMA. Only direction is returned: callers can
    retain the original demonstrated force magnitude.
    """
    if force.ndim != 3 or force.shape[-1] != 3:
        raise ValueError(f"force must have shape (batch, time, 3), got {tuple(force.shape)}")
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if valid_force_threshold < 0.0:
        raise ValueError("valid_force_threshold must be non-negative")

    if window == 1:
        force_smooth = force
    else:
        # Replication padding preserves sequence length without introducing zero
        # vectors at episode boundaries. avg_pool1d operates on (N, C, T).
        force_channels = force.transpose(1, 2)
        pad = window // 2
        force_smooth = F.avg_pool1d(
            F.pad(force_channels, (pad, pad), mode="replicate"),
            kernel_size=window,
            stride=1,
        ).transpose(1, 2)

    magnitude = torch.linalg.norm(force_smooth, dim=-1)
    candidate = force_smooth / magnitude.unsqueeze(-1).clamp(min=1.0e-6)
    valid = magnitude > float(valid_force_threshold)

    batch_size, sequence_length, _ = force.shape
    default = torch.zeros((batch_size, 3), dtype=force.dtype, device=force.device)
    default[:, 2] = 1.0
    has_valid = valid.any(dim=1)
    first_valid_idx = valid.long().argmax(dim=1)
    batch_idx = torch.arange(batch_size, device=force.device)
    direction = torch.where(has_valid.unsqueeze(-1), candidate[batch_idx, first_valid_idx], default)

    directions = torch.empty_like(force)
    for step in range(sequence_length):
        direction = torch.where(valid[:, step].unsqueeze(-1), candidate[:, step], direction)
        directions[:, step] = direction
    return directions


def construct_reference_plane_trajectory(
    reference_position: torch.Tensor,
    target_force: torch.Tensor,
    plane_stiffness: float | torch.Tensor,
    direction_smoothing_window: int,
    direction_force_threshold: float,
    power_law_exponent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct one virtual plane per reference timestep.

    The plane normal is the smoothed target-force direction. For outward normal
    ``n`` and plane point ``p``, penetration at reference position ``x_ref`` is
    ``relu(n dot (p - x_ref))``. Therefore choosing

    ``p = x_ref + (|F_goal| / k_p)**(1/p) * n``

    makes the elastic law ``k_p * penetration**p`` generate exactly
    ``|F_goal| * n`` when the end effector perfectly follows ``x_ref``. ``k_p`` is
    the fixed environment coefficient and is deliberately independent of the
    robot/controller stiffness. ``p=1`` recovers the linear construction.

    Returns ``(surface_point, surface_normal, aligned_target_force)``.
    """
    if reference_position.shape != target_force.shape or reference_position.shape[-1] != 3:
        raise ValueError(
            "reference_position and target_force must have matching (batch, time, 3) shapes; "
            f"got {tuple(reference_position.shape)} and {tuple(target_force.shape)}"
        )
    stiffness = torch.as_tensor(
        plane_stiffness, dtype=reference_position.dtype, device=reference_position.device
    )
    if torch.any(stiffness <= 0.0):
        raise ValueError("plane_stiffness must be positive")
    power_law_exponent = float(power_law_exponent)
    if power_law_exponent <= 0.0:
        raise ValueError("power_law_exponent must be positive")
    if stiffness.ndim == 0:
        stiffness = stiffness.view(1, 1, 1)
    elif stiffness.ndim == 1:
        stiffness = stiffness.view(-1, 1, 1)
    elif stiffness.ndim == 2:
        stiffness = stiffness.unsqueeze(1)

    normal = smooth_force_directions(
        target_force,
        window=direction_smoothing_window,
        valid_force_threshold=direction_force_threshold,
    )
    force_magnitude = torch.linalg.norm(target_force, dim=-1, keepdim=True)
    penetration = (force_magnitude / stiffness).pow(1.0 / power_law_exponent)
    surface_point = reference_position + penetration * normal
    aligned_target_force = force_magnitude * normal
    return surface_point, normal, aligned_target_force


def finite_difference_surface_velocity(
    surface_point: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Estimate per-timestep virtual-plane velocity from its point trajectory.

    Interior samples use centered differences; the first and last samples use
    one-sided differences. The resulting control-grid velocity can be held across
    all physics substeps belonging to that control step.
    """
    if surface_point.ndim != 3 or surface_point.shape[-1] != 3:
        raise ValueError(
            f"surface_point must have shape (batch, time, 3), got {tuple(surface_point.shape)}"
        )
    timestep = float(timestep)
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")

    velocity = torch.zeros_like(surface_point)
    sequence_length = surface_point.shape[1]
    if sequence_length <= 1:
        return velocity
    velocity[:, 0] = (surface_point[:, 1] - surface_point[:, 0]) / timestep
    velocity[:, -1] = (surface_point[:, -1] - surface_point[:, -2]) / timestep
    if sequence_length > 2:
        velocity[:, 1:-1] = (surface_point[:, 2:] - surface_point[:, :-2]) / (2.0 * timestep)
    return velocity


def compute_virtual_plane_contact(
    position: torch.Tensor,
    linear_velocity: torch.Tensor,
    surface_point: torch.Tensor,
    surface_normal: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    transition_width: float,
    force_cap: float | None,
    surface_velocity: torch.Tensor | None = None,
    contact_model: str = "linear",
    hunt_crossley_dissipation: float = 0.0,
    power_law_exponent: float = 1.5,
    exponential_sharpness: float = 2.0,
    exponential_reference_force: float = 5.0,
    exponential_reference_penetration: float = 0.002,
    max_damping_multiplier: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed plane distance and a selectable unilateral contact force.

    ``surface_normal`` points out of the solid half-space. Positive signed
    distance is free space; negative distance is penetration. Damping increases
    the reaction while moving into the surface and decreases it while separating,
    but is gated by penetration so it cannot create a pre-contact force.

    ``linear`` uses ``f_n = max(0, k*delta + c*delta_dot)``. The Drake-style
    ``drake_hunt_crossley`` model uses
    ``f_n = k*delta*max(0, 1 + d*delta_dot)``. Here
    ``delta_dot = n dot (v_surface - v_contact)`` is positive during compression.
    ``power_law`` replaces the elastic term with ``k_p*delta**p`` and optionally
    applies the same Hunt-Crossley multiplier. A positive
    ``max_damping_multiplier`` bounds that multiplier; ``None`` or a non-positive
    value disables this clamp. A non-positive ``force_cap`` disables force capping.
    """
    if contact_model not in ("linear", "drake_hunt_crossley", "power_law", "exponential"):
        raise ValueError(
            "contact_model must be 'linear', 'drake_hunt_crossley', 'power_law', or 'exponential', "
            f"got {contact_model!r}"
        )
    if hunt_crossley_dissipation < 0.0:
        raise ValueError("hunt_crossley_dissipation must be non-negative")
    power_law_exponent = float(power_law_exponent)
    if power_law_exponent <= 0.0:
        raise ValueError("power_law_exponent must be positive")
    exponential_sharpness = float(exponential_sharpness)
    exponential_reference_force = float(exponential_reference_force)
    exponential_reference_penetration = float(exponential_reference_penetration)
    if exponential_sharpness <= 0.0:
        raise ValueError("exponential_sharpness must be positive")
    if exponential_reference_force <= 0.0 or exponential_reference_penetration <= 0.0:
        raise ValueError("exponential reference force and penetration must be positive")
    if surface_velocity is None:
        surface_velocity = torch.zeros_like(linear_velocity)

    distance = ((position - surface_point) * surface_normal).sum(dim=-1, keepdim=True)
    penetration_raw = -distance
    penetration = torch.relu(penetration_raw)
    penetration_rate = ((surface_velocity - linear_velocity) * surface_normal).sum(
        dim=-1, keepdim=True
    )
    if contact_model == "power_law":
        elastic_force = stiffness * penetration.pow(power_law_exponent)
    elif contact_model == "exponential":
        exponent = (
            exponential_sharpness * penetration / exponential_reference_penetration
        ).clamp(max=50.0)
        elastic_force = exponential_reference_force * torch.expm1(exponent) / math.expm1(
            exponential_sharpness
        )
    else:
        elastic_force = stiffness * penetration
    if contact_model == "linear":
        force_magnitude = torch.relu(elastic_force + damping * penetration_rate)
    else:
        dissipation_factor = torch.relu(
            1.0 + float(hunt_crossley_dissipation) * penetration_rate
        )
        if max_damping_multiplier is not None and float(max_damping_multiplier) > 0.0:
            dissipation_factor = dissipation_factor.clamp(max=float(max_damping_multiplier))
        force_magnitude = elastic_force * dissipation_factor
    if transition_width > 0.0:
        transition = (penetration / float(transition_width)).clamp(0.0, 1.0)
        # C1-continuous smoothstep: zero value/slope at first contact and unit
        # value/zero slope once penetration reaches the transition width.
        activation = transition.square() * (3.0 - 2.0 * transition)
        force_magnitude = force_magnitude * activation
    force_magnitude = torch.where(
        penetration_raw > 0.0,
        force_magnitude,
        torch.zeros_like(force_magnitude),
    )
    if force_cap is not None and float(force_cap) > 0.0:
        force_magnitude = force_magnitude.clamp(max=float(force_cap))
    return distance, force_magnitude * surface_normal
