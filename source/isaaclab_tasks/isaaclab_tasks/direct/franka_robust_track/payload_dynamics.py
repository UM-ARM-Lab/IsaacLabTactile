"""Rigid-payload inertia utilities shared by replay and training."""

from __future__ import annotations

import torch


PAYLOAD_DYNAMICS_MODES = ("residual_wrench", "rigid_payload")


def validate_payload_dynamics_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in PAYLOAD_DYNAMICS_MODES:
        raise ValueError(
            f"payload_dynamics_mode must be one of {PAYLOAD_DYNAMICS_MODES}, "
            f"got {mode!r}"
        )
    return mode


def inertia_diag_from_shape(shape: torch.Tensor) -> torch.Tensor:
    """Convert nonnegative principal second moments to a physical diagonal.

    For shape ``[a, b, c]``, the principal inertia is
    ``[b+c, a+c, a+b]``. This enforces the rigid-body triangle inequalities
    without projecting optimizer samples.
    """
    if shape.shape[-1] != 3:
        raise ValueError(f"payload inertia shape must end in 3, got {shape.shape}")
    if bool((shape < 0.0).any()):
        raise ValueError("payload inertia shape must be nonnegative")
    a, b, c = shape.unbind(dim=-1)
    return torch.stack((b + c, a + c, a + b), dim=-1)


def inertia_shape_from_diag(diag: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`inertia_diag_from_shape` for a physical diagonal."""
    if diag.shape[-1] != 3:
        raise ValueError(f"payload inertia diagonal must end in 3, got {diag.shape}")
    ixx, iyy, izz = diag.unbind(dim=-1)
    shape = 0.5 * torch.stack(
        (iyy + izz - ixx, ixx + izz - iyy, ixx + iyy - izz), dim=-1
    )
    if bool((shape < -1.0e-9).any()):
        raise ValueError("payload inertia diagonal violates triangle inequalities")
    return shape.clamp_min(0.0)


def _parallel_axis(vector: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return (
        (vector * vector).sum(dim=-1)[..., None, None] * eye
        - vector.unsqueeze(-1) * vector.unsqueeze(-2)
    )


def replace_payload_component(
    masses: torch.Tensor,
    coms: torch.Tensor,
    inertias: torch.Tensor,
    *,
    body_idx: int,
    env_ids: torch.Tensor,
    nominal_mass: torch.Tensor,
    nominal_com: torch.Tensor,
    nominal_inertia_diag: torch.Tensor,
    actual_mass: torch.Tensor,
    actual_first_moment: torch.Tensor,
    actual_inertia_shape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace one declared payload component in articulation inertials.

    The articulation body may also contain an adapter or base hand. We subtract
    the declared nominal payload in spatial-inertia coordinates about the body
    origin, then add the candidate physical payload. ``masses``, ``coms``, and
    ``inertias`` are updated in place. Inertias are about each component's CoM
    and expressed in the payload-body frame.
    """
    env_ids = env_ids.to(device=masses.device, dtype=torch.long)
    count = len(env_ids)
    dtype, device = masses.dtype, masses.device

    def rows(value, width):
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        return tensor.reshape(-1, width).expand(count, width)

    nominal_mass = rows(nominal_mass, 1).squeeze(-1)
    nominal_com = rows(nominal_com, 3)
    nominal_inertia_diag = rows(nominal_inertia_diag, 3)
    actual_mass = rows(actual_mass, 1).squeeze(-1)
    actual_first_moment = rows(actual_first_moment, 3)
    actual_inertia_diag = inertia_diag_from_shape(rows(actual_inertia_shape, 3))

    if bool((actual_mass <= 0.0).any()):
        raise ValueError("rigid payload mass must be positive")
    actual_com = actual_first_moment / actual_mass.unsqueeze(-1)

    body_mass = masses[env_ids, body_idx]
    body_com = coms[env_ids, body_idx, :3]
    body_inertia = inertias[env_ids, body_idx].reshape(-1, 3, 3)
    body_first_moment = body_mass.unsqueeze(-1) * body_com
    body_inertia_origin = body_inertia + body_mass[:, None, None] * _parallel_axis(body_com)

    nominal_inertia = torch.diag_embed(nominal_inertia_diag)
    nominal_first_moment = nominal_mass.unsqueeze(-1) * nominal_com
    nominal_inertia_origin = (
        nominal_inertia
        + nominal_mass[:, None, None] * _parallel_axis(nominal_com)
    )

    base_mass = body_mass - nominal_mass
    base_first_moment = body_first_moment - nominal_first_moment
    base_inertia_origin = body_inertia_origin - nominal_inertia_origin

    actual_inertia = torch.diag_embed(actual_inertia_diag)
    actual_inertia_origin = (
        actual_inertia
        + actual_mass[:, None, None] * _parallel_axis(actual_com)
    )
    combined_mass = base_mass + actual_mass
    combined_first_moment = base_first_moment + actual_first_moment
    combined_com = combined_first_moment / combined_mass.unsqueeze(-1)
    combined_inertia = (
        base_inertia_origin
        + actual_inertia_origin
        - combined_mass[:, None, None] * _parallel_axis(combined_com)
    )
    combined_inertia = 0.5 * (
        combined_inertia + combined_inertia.transpose(-1, -2)
    )
    if bool((combined_mass <= 0.0).any()):
        raise ValueError("rigid payload replacement produced nonpositive body mass")
    if bool((torch.linalg.eigvalsh(combined_inertia) <= 0.0).any()):
        raise ValueError("rigid payload replacement produced invalid body inertia")

    masses[env_ids, body_idx] = combined_mass
    coms[env_ids, body_idx, :3] = combined_com
    inertias[env_ids, body_idx] = combined_inertia.reshape(-1, 9)
    return actual_com, actual_inertia_diag


def residual_gravity_parameters(
    actual_mass: torch.Tensor,
    actual_first_moment: torch.Tensor,
    desk_mass: float,
    desk_com,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mass and first-moment errors left after Desk compensation."""
    desk_com = torch.as_tensor(
        desk_com, device=actual_first_moment.device, dtype=actual_first_moment.dtype
    )
    return (
        actual_mass - float(desk_mass),
        actual_first_moment - float(desk_mass) * desk_com,
    )
