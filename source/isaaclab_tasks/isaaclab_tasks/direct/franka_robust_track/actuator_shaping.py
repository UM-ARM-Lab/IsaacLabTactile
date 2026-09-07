"""libfranka-compatible torque filtering and rate limiting."""

from __future__ import annotations

import math

import torch


def _shape_tick(command, desired, gain, max_delta):
    filtered = gain * command + (1.0 - gain) * desired
    return desired + torch.clamp(filtered - desired, -max_delta, max_delta)


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
        desired = _shape_tick(command, desired, gain, max_delta)
        total += desired
    return total / ticks, desired


def shape_torque_over_interval(
    command: torch.Tensor,
    previous_desired: torch.Tensor,
    *,
    interval_dt: float,
    time_to_next_tick: float = 0.0,
    hardware_dt: float = 0.001,
    cutoff_hz: float = 100.0,
    max_rate: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Integrate a held torque command across asynchronous hardware ticks.

    The last hardware output remains held across a physics boundary until the
    next hardware tick. Return the time-weighted mean, final output and phase.
    """
    gain = hardware_dt / (hardware_dt + 1.0 / (2.0 * math.pi * cutoff_hz))
    desired = previous_desired
    total = torch.zeros_like(desired)
    remaining = interval_dt
    wait = time_to_next_tick
    while remaining > 1.0e-12:
        if wait <= 1.0e-12:
            desired = _shape_tick(command, desired, gain, max_rate * hardware_dt)
            wait = hardware_dt
        duration = min(wait, remaining)
        total += desired * duration
        remaining -= duration
        wait -= duration
    return total / interval_dt, desired, max(0.0, wait)
