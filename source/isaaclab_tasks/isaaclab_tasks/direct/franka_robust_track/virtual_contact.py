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


def x_axis_to_vector_quat(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return ``wxyz`` quaternions that rotate local +x onto ``vector``.

    Zero-length vectors receive the identity rotation. The helper is used by the
    debug visualization for both force arrows and the virtual plane, whose local
    thin axis is +x.
    """
    if vector.shape[-1] != 3:
        raise ValueError(
            "vector must have trailing dimension 3, "
            f"got {tuple(vector.shape)}"
        )
    magnitude = torch.linalg.norm(vector, dim=-1, keepdim=True)
    direction = vector / magnitude.clamp_min(eps)

    # For source axis a=(1,0,0), the shortest-arc quaternion to unit b is
    # normalize([1 + dot(a,b), cross(a,b)]). Handle b=-a separately because
    # that expression is identically zero; +180 degrees around z maps +x to -x.
    quat = torch.stack(
        (
            1.0 + direction[..., 0],
            torch.zeros_like(direction[..., 0]),
            -direction[..., 2],
            direction[..., 1],
        ),
        dim=-1,
    )
    opposite = direction[..., 0] < (-1.0 + 1.0e-6)
    opposite_quat = torch.zeros_like(quat)
    opposite_quat[..., 3] = 1.0
    quat = torch.where(opposite.unsqueeze(-1), opposite_quat, quat)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)

    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where((magnitude <= eps).expand_as(quat), identity, quat)


def z_axis_to_vector_quat(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return ``wxyz`` quaternions that rotate local +z onto ``vector``.

    Isaac Sim 5.0's UI arrow asset is visually rooted along local +z, despite
    being distributed as ``arrow_x.usd``. This transform is therefore used for
    arrow markers, while :func:`x_axis_to_vector_quat` remains correct for the
    virtual-plane cuboid's thin local +x axis.
    """
    if vector.shape[-1] != 3:
        raise ValueError(
            "vector must have trailing dimension 3, "
            f"got {tuple(vector.shape)}"
        )
    magnitude = torch.linalg.norm(vector, dim=-1, keepdim=True)
    direction = vector / magnitude.clamp_min(eps)

    # For source axis a=(0,0,1), construct the shortest-arc quaternion
    # normalize([1 + dot(a,b), cross(a,b)]). Use +180 degrees around x for -z.
    quat = torch.stack(
        (
            1.0 + direction[..., 2],
            -direction[..., 1],
            direction[..., 0],
            torch.zeros_like(direction[..., 0]),
        ),
        dim=-1,
    )
    opposite = direction[..., 2] < (-1.0 + 1.0e-6)
    opposite_quat = torch.zeros_like(quat)
    opposite_quat[..., 1] = 1.0
    quat = torch.where(opposite.unsqueeze(-1), opposite_quat, quat)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)

    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where((magnitude <= eps).expand_as(quat), identity, quat)


def world_force_components(force_world: torch.Tensor) -> torch.Tensor:
    """Split a world force into signed x/y/z vectors whose sum is the force."""
    if force_world.shape[-1] != 3:
        raise ValueError(
            "force_world must have trailing dimension 3, "
            f"got {tuple(force_world.shape)}"
        )
    return torch.diag_embed(force_world)


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
    torque_scale: float = 1.0,
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
    torque_scale = float(torque_scale)
    if torque_scale < 0.0:
        raise ValueError("torque_scale must be non-negative")
    torque = torque_scale * target_torque_world * ratio

    if torque_cap is not None and float(torque_cap) > 0.0:
        magnitude = torch.linalg.norm(torque, dim=-1, keepdim=True)
        torque = torque * (float(torque_cap) / magnitude.clamp(min=1.0e-6)).clamp(max=1.0)
    return torque


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions with broadcast-compatible leading shapes."""
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_inverse_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_from_rotation_vector(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half_angle) / angle.clamp_min(1.0e-12),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half_angle), scale * rotation_vector), dim=-1)


def _rotation_vector_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    # q and -q encode the same orientation. Select the shortest rotation.
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    vector = quat[..., 1:]
    vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., :1].clamp_min(1.0e-12))
    scale = torch.where(
        vector_norm > 1.0e-7,
        angle / vector_norm.clamp_min(1.0e-12),
        2.0 + vector_norm.square() / 3.0,
    )
    return scale * vector


def _contact_parameter(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    name: str,
    *,
    strictly_positive: bool = False,
) -> torch.Tensor:
    """Convert a scalar or per-environment contact parameter for broadcasting."""
    parameter = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if parameter.ndim > 0:
        if parameter.shape[0] != reference.shape[0]:
            raise ValueError(
                f"{name} first dimension must match the batch size "
                f"{reference.shape[0]}, got {tuple(parameter.shape)}"
            )
        if parameter.ndim == 1:
            parameter = parameter.unsqueeze(-1)
        while parameter.ndim < reference.ndim:
            parameter = parameter.unsqueeze(1)
    invalid = parameter <= 0.0 if strictly_positive else parameter < 0.0
    if torch.any(invalid):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return parameter


def perturb_reference_plane_trajectory(
    surface_point: torch.Tensor,
    surface_normal: torch.Tensor,
    normal_offset: float | torch.Tensor,
    tilt_rotation_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one hidden plane offset and normal rotation per environment.

    ``normal_offset`` is a signed distance along the perturbed normal. Positive
    values move the surface toward the reference pose's solid half-space and
    therefore increase penetration. ``tilt_rotation_vector`` is an axis-angle
    world-frame rotation held constant over the complete reference trajectory.
    """
    if surface_point.shape != surface_normal.shape or surface_point.ndim != 3:
        raise ValueError(
            "surface_point and surface_normal must have matching (batch, time, 3) "
            f"shapes, got {tuple(surface_point.shape)} and {tuple(surface_normal.shape)}"
        )
    if surface_point.shape[-1] != 3:
        raise ValueError("surface trajectories must have a trailing xyz dimension")
    if tilt_rotation_vector.shape != (surface_point.shape[0], 3):
        raise ValueError(
            "tilt_rotation_vector must have shape (batch, 3), got "
            f"{tuple(tilt_rotation_vector.shape)}"
        )
    offset = torch.as_tensor(
        normal_offset, dtype=surface_point.dtype, device=surface_point.device
    )
    if offset.ndim > 0 and offset.shape[0] != surface_point.shape[0]:
        raise ValueError(
            "normal_offset first dimension must match the batch size "
            f"{surface_point.shape[0]}, got {tuple(offset.shape)}"
        )
    if offset.ndim == 0:
        offset = offset.reshape(1, 1, 1)
    elif offset.ndim == 1:
        offset = offset.unsqueeze(-1)
    while offset.ndim < surface_point.ndim:
        offset = offset.unsqueeze(1)

    tilt_quat = _quat_from_rotation_vector(tilt_rotation_vector).unsqueeze(1).expand(
        -1, surface_normal.shape[1], -1
    )
    perturbed_normal = _quat_apply_wxyz(tilt_quat, surface_normal)
    perturbed_normal = torch.nn.functional.normalize(perturbed_normal, dim=-1)
    perturbed_point = surface_point + offset * perturbed_normal
    return perturbed_point, perturbed_normal


def construct_rotational_equilibrium_trajectory(
    reference_quat_world: torch.Tensor,
    target_torque_world: torch.Tensor,
    rotational_stiffness: float | torch.Tensor,
    power_law_exponent: float | torch.Tensor = 1.0,
    torque_scale: float = 1.0,
    torque_cap: float | None = None,
) -> torch.Tensor:
    """Construct an orientation equilibrium that reproduces target torque.

    The rotational spring uses ``|tau| = K_rot * |theta|**p`` in world axes.
    The default ``p=1`` recovers the linear spring.  The equilibrium offset is
    chosen by inverting the selected law, so the demonstrated orientation
    produces ``tau_target`` at zero relative angular velocity.
    """
    torque_scale = float(torque_scale)
    if torque_scale <= 0.0:
        raise ValueError("torque_scale must be positive")
    if reference_quat_world.shape[-1] != 4 or target_torque_world.shape[-1] != 3:
        raise ValueError("reference quaternion and target torque must end in 4 and 3")
    if reference_quat_world.shape[:-1] != target_torque_world.shape[:-1]:
        raise ValueError("reference quaternion and target torque leading shapes must match")
    realizable_target_torque = target_torque_world
    if torque_cap is not None and float(torque_cap) > 0.0:
        # The spring is scaled before the physical cap. Limit its equilibrium
        # offset to the largest raw torque that can actually reach the robot.
        raw_cap = float(torque_cap) / torque_scale
        magnitude = torch.linalg.norm(
            realizable_target_torque, dim=-1, keepdim=True
        )
        realizable_target_torque = realizable_target_torque * (
            raw_cap / magnitude.clamp_min(1.0e-6)
        ).clamp(max=1.0)
    target_magnitude = torch.linalg.norm(
        realizable_target_torque, dim=-1, keepdim=True
    )
    rotational_stiffness = _contact_parameter(
        rotational_stiffness,
        target_magnitude,
        "rotational_stiffness",
        strictly_positive=True,
    )
    power_law_exponent = _contact_parameter(
        power_law_exponent,
        target_magnitude,
        "power_law_exponent",
        strictly_positive=True,
    )
    target_direction = realizable_target_torque / target_magnitude.clamp_min(1.0e-12)
    offset_angle = (target_magnitude / rotational_stiffness).pow(
        1.0 / power_law_exponent
    )
    offset = _quat_from_rotation_vector(target_direction * offset_angle)
    return torch.nn.functional.normalize(
        _quat_mul_wxyz(offset, reference_quat_world), dim=-1
    )


def finite_difference_quaternion_angular_velocity(
    quat_world: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Estimate world angular velocity from a (batch, time, 4) trajectory."""
    if quat_world.ndim != 3 or quat_world.shape[-1] != 4:
        raise ValueError(f"quat_world must have shape (batch, time, 4), got {tuple(quat_world.shape)}")
    timestep = float(timestep)
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    velocity = torch.zeros((*quat_world.shape[:-1], 3), device=quat_world.device, dtype=quat_world.dtype)
    if quat_world.shape[1] <= 1:
        return velocity
    interval_velocity = _rotation_vector_from_quat(
        _quat_mul_wxyz(quat_world[:, 1:], _quat_inverse_wxyz(quat_world[:, :-1]))
    ) / timestep
    velocity[:, 0] = interval_velocity[:, 0]
    velocity[:, -1] = interval_velocity[:, -1]
    if quat_world.shape[1] > 2:
        velocity[:, 1:-1] = 0.5 * (
            interval_velocity[:, :-1] + interval_velocity[:, 1:]
        )
    return velocity


def quaternion_nlerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Shortest-path normalized quaternion interpolation."""
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
    return torch.nn.functional.normalize(q0 + blend * (q1 - q0), dim=-1)


def orientation_spring_contact_torque(
    current_quat_world: torch.Tensor,
    current_angvel_world: torch.Tensor,
    equilibrium_quat_world: torch.Tensor,
    equilibrium_angvel_world: torch.Tensor,
    in_contact: torch.Tensor,
    rotational_stiffness: float | torch.Tensor,
    rotational_damping: float | torch.Tensor,
    torque_scale: float = 1.0,
    torque_cap: float | None = None,
) -> torch.Tensor:
    """Rotational Kelvin-Voigt contact about a moving equilibrium orientation."""
    torque_scale = float(torque_scale)
    if torque_scale < 0.0:
        raise ValueError("torque_scale must be non-negative")
    error_quat = _quat_mul_wxyz(
        equilibrium_quat_world,
        _quat_inverse_wxyz(current_quat_world),
    )
    orientation_error = _rotation_vector_from_quat(error_quat)
    relative_angvel = equilibrium_angvel_world - current_angvel_world
    rotational_stiffness = _contact_parameter(
        rotational_stiffness,
        orientation_error[..., :1],
        "rotational_stiffness",
        strictly_positive=True,
    )
    rotational_damping = _contact_parameter(
        rotational_damping,
        orientation_error[..., :1],
        "rotational_damping",
    )
    torque = torque_scale * (
        rotational_stiffness * orientation_error
        + rotational_damping * relative_angvel
    )
    torque = torch.where(in_contact.unsqueeze(-1), torque, torch.zeros_like(torque))
    if torque_cap is not None and float(torque_cap) > 0.0:
        magnitude = torch.linalg.norm(torque, dim=-1, keepdim=True)
        torque = torque * (float(torque_cap) / magnitude.clamp_min(1.0e-6)).clamp(max=1.0)
    return torque


def orientation_power_law_contact_torque(
    current_quat_world: torch.Tensor,
    current_angvel_world: torch.Tensor,
    equilibrium_quat_world: torch.Tensor,
    equilibrium_angvel_world: torch.Tensor,
    in_contact: torch.Tensor,
    rotational_coefficient: float | torch.Tensor,
    power_law_exponent: float | torch.Tensor,
    rotational_damping: float | torch.Tensor,
    torque_scale: float = 1.0,
    torque_cap: float | None = None,
) -> torch.Tensor:
    """Isotropic power-law rotational contact about a moving equilibrium.

    For the rotation vector ``theta = Log(R_eq R_current^-1)``, the elastic
    moment is ``K_p * |theta|**(p-1) * theta``.  This is the orientation-space
    analogue of the translational normal law ``f = k_p * delta**p``.  Viscous
    damping remains additive in angular velocity so it is well defined at zero
    angular displacement.
    """
    torque_scale = float(torque_scale)
    if torque_scale < 0.0:
        raise ValueError("torque_scale must be non-negative")

    error_quat = _quat_mul_wxyz(
        equilibrium_quat_world,
        _quat_inverse_wxyz(current_quat_world),
    )
    orientation_error = _rotation_vector_from_quat(error_quat)
    error_magnitude = torch.linalg.norm(orientation_error, dim=-1, keepdim=True)
    rotational_coefficient = _contact_parameter(
        rotational_coefficient,
        error_magnitude,
        "rotational_coefficient",
        strictly_positive=True,
    )
    power_law_exponent = _contact_parameter(
        power_law_exponent,
        error_magnitude,
        "power_law_exponent",
        strictly_positive=True,
    )
    rotational_damping = _contact_parameter(
        rotational_damping,
        error_magnitude,
        "rotational_damping",
    )
    elastic_torque = (
        rotational_coefficient
        * error_magnitude.pow(power_law_exponent)
        * orientation_error
        / error_magnitude.clamp_min(1.0e-12)
    )
    relative_angvel = equilibrium_angvel_world - current_angvel_world
    torque = torque_scale * (
        elastic_torque + rotational_damping * relative_angvel
    )
    torque = torch.where(in_contact.unsqueeze(-1), torque, torch.zeros_like(torque))
    if torque_cap is not None and float(torque_cap) > 0.0:
        magnitude = torch.linalg.norm(torque, dim=-1, keepdim=True)
        torque = torque * (
            float(torque_cap) / magnitude.clamp_min(1.0e-6)
        ).clamp(max=1.0)
    return torque


def descale_virtual_sensor_wrench(
    sensor_wrench: torch.Tensor,
    force_scale: float,
    torque_scale: float | None = None,
) -> torch.Tensor:
    """Map a physically scaled virtual-contact wrench back to policy units.

    Force and torque can have independent physical scales. Omitting
    ``torque_scale`` retains the legacy behavior of dividing the entire wrench
    by ``force_scale``.
    """
    force_scale = float(force_scale)
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    if sensor_wrench.shape[-1] == 3:
        return sensor_wrench / force_scale
    if sensor_wrench.shape[-1] != 6:
        raise ValueError(
            "sensor_wrench must have trailing dimension 3 or 6, "
            f"got {tuple(sensor_wrench.shape)}"
        )
    if torque_scale is None:
        torque_scale = force_scale
    torque_scale = float(torque_scale)
    if torque_scale <= 0.0:
        raise ValueError("torque_scale must be positive")
    return torch.cat(
        (
            sensor_wrench[..., :3] / force_scale,
            sensor_wrench[..., 3:] / torque_scale,
        ),
        dim=-1,
    )


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
    reference_force: float | torch.Tensor,
    reference_penetration: float | torch.Tensor,
    exponent: float | torch.Tensor,
) -> float | torch.Tensor:
    """Return ``k_p = f_ref / delta_ref**p`` for a power-law contact."""
    if not any(
        isinstance(value, torch.Tensor)
        for value in (reference_force, reference_penetration, exponent)
    ):
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

    tensor = next(
        value for value in (reference_force, reference_penetration, exponent)
        if isinstance(value, torch.Tensor)
    )
    reference_force = torch.as_tensor(reference_force, dtype=tensor.dtype, device=tensor.device)
    reference_penetration = torch.as_tensor(
        reference_penetration, dtype=tensor.dtype, device=tensor.device
    )
    exponent = torch.as_tensor(exponent, dtype=tensor.dtype, device=tensor.device)
    if torch.any(reference_force <= 0.0):
        raise ValueError("reference_force must be positive")
    if torch.any(reference_penetration <= 0.0):
        raise ValueError("reference_penetration must be positive")
    if torch.any(exponent <= 0.0):
        raise ValueError("exponent must be positive")
    return reference_force / reference_penetration.pow(exponent)


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
    power_law_exponent: float | torch.Tensor = 1.0,
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
    parameter_reference = reference_position[..., :1]
    stiffness = _contact_parameter(
        plane_stiffness,
        parameter_reference,
        "plane_stiffness",
        strictly_positive=True,
    )
    power_law_exponent = _contact_parameter(
        power_law_exponent,
        parameter_reference,
        "power_law_exponent",
        strictly_positive=True,
    )

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
    hunt_crossley_dissipation: float | torch.Tensor = 0.0,
    power_law_exponent: float | torch.Tensor = 1.5,
    exponential_sharpness: float = 2.0,
    exponential_reference_force: float = 5.0,
    exponential_reference_penetration: float = 0.002,
    max_damping_multiplier: float | None = None,
    force_scale: float = 1.0,
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
    value disables this clamp. ``force_scale`` multiplies the modeled force
    before the final cap. A non-positive ``force_cap`` disables force capping.
    """
    if contact_model not in ("linear", "drake_hunt_crossley", "power_law", "exponential"):
        raise ValueError(
            "contact_model must be 'linear', 'drake_hunt_crossley', 'power_law', or 'exponential', "
            f"got {contact_model!r}"
        )
    force_scale = float(force_scale)
    if force_scale < 0.0:
        raise ValueError("force_scale must be non-negative")
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
    power_law_exponent = _contact_parameter(
        power_law_exponent,
        penetration,
        "power_law_exponent",
        strictly_positive=True,
    )
    hunt_crossley_dissipation = _contact_parameter(
        hunt_crossley_dissipation,
        penetration,
        "hunt_crossley_dissipation",
    )
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
            1.0 + hunt_crossley_dissipation * penetration_rate
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
    force_magnitude = force_scale * force_magnitude
    if force_cap is not None and float(force_cap) > 0.0:
        force_magnitude = force_magnitude.clamp(max=float(force_cap))
    return distance, force_magnitude * surface_normal
