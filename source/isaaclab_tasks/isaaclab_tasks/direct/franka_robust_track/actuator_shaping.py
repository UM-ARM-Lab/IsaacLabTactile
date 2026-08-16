"""libfranka-compatible torque filtering and rate limiting."""

from __future__ import annotations

import math

import torch


def shape_torque_held_command(
    command: torch.Tensor,
    previous_desired: torch.Tensor,
    *,
    ticks: int,
    hardware_dt: float = 0.001,
    cutoff_hz: float = 100.0,
    max_rate: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return interval-average torque and final tau_J_d for a held command."""
    gain = hardware_dt / (hardware_dt + 1.0 / (2.0 * math.pi * cutoff_hz))
    max_delta = max_rate * hardware_dt
    desired = previous_desired
    total = torch.zeros_like(desired)
    for _ in range(ticks):
        filtered = gain * command + (1.0 - gain) * desired
        desired = desired + torch.clamp(filtered - desired, -max_delta, max_delta)
        total += desired
    return total / ticks, desired
