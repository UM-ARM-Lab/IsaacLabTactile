"""Pure dual-clock indexing helpers for robust trajectory tracking."""

from __future__ import annotations

import math

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


def dataset_reference_end_timeouts(
    policy_steps: torch.Tensor,
    episode_lengths: torch.Tensor,
    episode_starts: torch.Tensor,
    *,
    chunk_length: int,
    policy_steps_per_reference: int,
) -> torch.Tensor:
    """Return per-environment timeouts at the last valid dataset reference.

    ``episode_lengths`` and ``episode_starts`` are native-reference indices for
    each active environment. ``chunk_length`` is a maximum: a chunk sampled near
    the end of a demonstration terminates early instead of executing repeated
    terminal padding. The timeout is expressed at the policy-clock instant of
    the last valid native sample.
    """
    chunk_length = int(chunk_length)
    policy_steps_per_reference = int(policy_steps_per_reference)
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    if policy_steps_per_reference < 1:
        raise ValueError("policy_steps_per_reference must be positive")
    if not (
        policy_steps.shape == episode_lengths.shape == episode_starts.shape
    ):
        raise ValueError(
            "policy_steps, episode_lengths, and episode_starts must have the same shape"
        )
    remaining_reference_steps = episode_lengths - episode_starts
    valid_reference_steps = remaining_reference_steps.clamp(
        min=1, max=chunk_length
    )
    last_valid_policy_step = (
        valid_reference_steps - 1
    ) * policy_steps_per_reference
    return policy_steps >= last_valid_policy_step


def valid_step_weighted_start_times(
    unit_samples: torch.Tensor,
    *,
    start_time_lo: float,
    start_time_hi: float,
    reference_duration: float,
    episode_duration: float,
) -> torch.Tensor:
    """Map uniform samples to start times weighted by valid episode duration.

    A zero-width range is an intentional fixed start, notably the phase-zero
    common evaluation. It must remain valid even when training uses weighted
    padding for non-degenerate start ranges.
    """
    start_time_lo = float(start_time_lo)
    start_time_hi = float(start_time_hi)
    reference_duration = float(reference_duration)
    episode_duration = float(episode_duration)
    if start_time_lo > start_time_hi:
        raise ValueError("start_time_lo must not exceed start_time_hi")
    if episode_duration <= 0.0:
        raise ValueError("episode_duration must be positive")
    if start_time_lo == start_time_hi:
        return torch.full_like(unit_samples, start_time_lo)

    weighted_hi = min(start_time_hi, reference_duration)
    weighted_lo = start_time_lo
    if weighted_lo >= weighted_hi:
        raise ValueError(
            "start-time range must overlap times before reference_duration"
        )

    full_valid_hi = min(
        weighted_hi,
        max(weighted_lo, reference_duration - episode_duration),
    )
    full_area = episode_duration * (full_valid_hi - weighted_lo)
    tail_lo = max(weighted_lo, reference_duration - episode_duration)
    tail_remaining_lo = reference_duration - tail_lo
    tail_remaining_hi = reference_duration - weighted_hi
    tail_area = 0.5 * (tail_remaining_lo**2 - tail_remaining_hi**2)
    total_area = full_area + tail_area
    area_sample = total_area * unit_samples

    uniform_time = weighted_lo + area_sample / episode_duration
    tail_area_sample = torch.clamp(area_sample - full_area, min=0.0)
    tail_time = reference_duration - torch.sqrt(
        torch.clamp(
            tail_remaining_lo**2 - 2.0 * tail_area_sample,
            min=0.0,
        )
    )
    return torch.where(area_sample < full_area, uniform_time, tail_time)


def phase_bin_indices(
    phase_times: torch.Tensor, *, bin_size_s: float, num_bins: int
) -> torch.Tensor:
    """Map reference times to fixed temporal bins, including the terminal time."""
    bin_size_s = float(bin_size_s)
    num_bins = int(num_bins)
    if bin_size_s <= 0.0:
        raise ValueError("bin_size_s must be positive")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    return torch.floor(phase_times / bin_size_s).long().clamp(0, num_bins - 1)


def adaptive_bin_sampling_probabilities(
    failure_counts: torch.Tensor,
    visit_counts: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    uniform_rate: float = 0.1,
    failure_rate_max_over_mean: float = 50.0,
    max_probability_multiplier: float = 50.0,
) -> torch.Tensor:
    """Mix failure-rate sampling with a weighted non-adaptive floor."""
    if failure_counts.ndim != 1 or visit_counts.shape != failure_counts.shape:
        raise ValueError("failure_counts and visit_counts must be matching vectors")
    if base_weights.shape != failure_counts.shape:
        raise ValueError("base_weights must match the count vectors")
    if not all(
        torch.is_floating_point(value)
        for value in (failure_counts, visit_counts, base_weights)
    ):
        raise ValueError("counts and base_weights must be floating point")
    if torch.any(failure_counts < 0.0) or torch.any(visit_counts <= 0.0):
        raise ValueError("counts must be nonnegative and visits must be positive")
    if torch.any(base_weights < 0.0) or base_weights.sum() <= 0.0:
        raise ValueError("base_weights must be nonnegative with positive sum")

    uniform_rate = float(uniform_rate)
    failure_rate_max_over_mean = float(failure_rate_max_over_mean)
    max_probability_multiplier = float(max_probability_multiplier)
    if not 0.0 <= uniform_rate <= 1.0:
        raise ValueError("uniform_rate must be in [0, 1]")
    if failure_rate_max_over_mean <= 0.0:
        raise ValueError("failure_rate_max_over_mean must be positive")
    if max_probability_multiplier <= 0.0:
        raise ValueError("max_probability_multiplier must be positive")

    base_prob = base_weights.double()
    base_prob /= base_prob.sum()
    failure_rate = failure_counts.double() / visit_counts.double()
    active = base_prob > 0.0
    active_mean = failure_rate[active].mean()
    clipped_rate = failure_rate.clamp(
        min=0.0, max=active_mean * failure_rate_max_over_mean
    )
    failure_mass = clipped_rate * base_prob
    if failure_mass.sum() > 0.0:
        failure_prob = failure_mass / failure_mass.sum()
    else:
        failure_prob = base_prob
    probabilities = (
        (1.0 - uniform_rate) * failure_prob + uniform_rate * base_prob
    )

    max_probability = max_probability_multiplier * base_prob
    probabilities = torch.minimum(probabilities, max_probability)
    probabilities /= probabilities.sum()
    return probabilities.to(dtype=failure_counts.dtype)


def variable_reference_bin_layout(
    reference_lengths: torch.Tensor,
    *,
    bin_size_steps: int,
    sequence_length_agnostic: bool = True,
    chunk_length: int = 0,
    valid_step_weighting: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SONIC-style local-frame bins for variable-length references.

    Bins are flattened across sequences but retain their sequence id and local
    frame bounds. With ``sequence_length_agnostic``, every bin is divided by
    the number of bins in its sequence, matching SONIC's variable-clip base
    weighting. Optional valid-step weighting preserves RobustTrack's legacy
    preference for starts that leave more executable (non-padded) targets.
    """
    if reference_lengths.ndim != 1 or reference_lengths.numel() < 1:
        raise ValueError("reference_lengths must be a non-empty vector")
    if reference_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("reference_lengths must be an integer tensor")
    if torch.any(reference_lengths <= 0):
        raise ValueError("reference_lengths must be positive")
    bin_size_steps = int(bin_size_steps)
    chunk_length = int(chunk_length)
    if bin_size_steps < 1:
        raise ValueError("bin_size_steps must be positive")
    if chunk_length < 0:
        raise ValueError("chunk_length must be nonnegative")
    if valid_step_weighting and chunk_length < 1:
        raise ValueError("valid_step_weighting requires a positive chunk_length")

    device = reference_lengths.device
    bins_per_sequence = torch.div(
        reference_lengths + bin_size_steps - 1,
        bin_size_steps,
        rounding_mode="floor",
    )
    bin_offsets = torch.cumsum(bins_per_sequence, dim=0) - bins_per_sequence
    sequence_ids = torch.repeat_interleave(
        torch.arange(reference_lengths.numel(), device=device), bins_per_sequence
    )
    repeated_offsets = torch.repeat_interleave(bin_offsets, bins_per_sequence)
    local_bin_ids = (
        torch.arange(sequence_ids.numel(), device=device) - repeated_offsets
    )
    bin_starts = local_bin_ids * bin_size_steps
    bin_ends = torch.minimum(
        bin_starts + bin_size_steps, reference_lengths[sequence_ids]
    )
    bin_lengths = bin_ends - bin_starts
    base_weights = bin_lengths.float()
    if sequence_length_agnostic:
        base_weights /= bins_per_sequence[sequence_ids].float()

    if valid_step_weighting:
        sequence_lengths = reference_lengths[sequence_ids]
        # Exact mean of min(chunk_length, sequence_length - start) over every
        # integer start in each bin, expressed as a fraction of chunk_length.
        full_end = torch.minimum(
            bin_ends,
            (sequence_lengths - chunk_length + 1).clamp_min(0),
        )
        num_full = (full_end - bin_starts).clamp(min=0)
        tail_start = bin_starts + num_full
        num_tail = bin_ends - tail_start
        tail_index_sum = (
            (tail_start + bin_ends - 1).double() * num_tail.double() * 0.5
        )
        valid_step_sum = (
            num_full.double() * chunk_length
            + num_tail.double() * sequence_lengths.double()
            - tail_index_sum
        )
        valid_fraction = valid_step_sum / (
            bin_lengths.double() * chunk_length
        )
        base_weights *= valid_fraction.to(base_weights.dtype)

    return (
        bin_offsets,
        bins_per_sequence,
        sequence_ids,
        bin_starts,
        bin_ends,
        base_weights,
    )


def variable_reference_bin_indices(
    sequence_ids: torch.Tensor,
    reference_steps: torch.Tensor,
    *,
    bin_offsets: torch.Tensor,
    bins_per_sequence: torch.Tensor,
    bin_size_steps: int,
) -> torch.Tensor:
    """Map ``(sequence id, local frame)`` pairs to flattened bin indices."""
    if sequence_ids.shape != reference_steps.shape:
        raise ValueError("sequence_ids and reference_steps must have matching shapes")
    if bin_offsets.ndim != 1 or bins_per_sequence.shape != bin_offsets.shape:
        raise ValueError("bin offsets and counts must be matching vectors")
    if int(bin_size_steps) < 1:
        raise ValueError("bin_size_steps must be positive")
    local_bins = torch.div(
        reference_steps.clamp_min(0), int(bin_size_steps), rounding_mode="floor"
    )
    local_bins = torch.minimum(
        local_bins, bins_per_sequence[sequence_ids] - 1
    )
    return bin_offsets[sequence_ids] + local_bins


def sample_variable_reference_starts(
    probabilities: torch.Tensor,
    bin_sequence_ids: torch.Tensor,
    bin_starts: torch.Tensor,
    bin_ends: torch.Tensor,
    *,
    num_samples: int,
    pre_failure_window_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a sequence and local start frame from flattened adaptive bins."""
    if probabilities.ndim != 1 or probabilities.numel() < 1:
        raise ValueError("probabilities must be a non-empty vector")
    if not (
        bin_sequence_ids.shape
        == bin_starts.shape
        == bin_ends.shape
        == probabilities.shape
    ):
        raise ValueError("bin metadata must match probabilities")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if pre_failure_window_steps < 0:
        raise ValueError("pre_failure_window_steps must be nonnegative")
    if torch.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("probabilities must be nonnegative with positive sum")
    if torch.any(bin_ends <= bin_starts):
        raise ValueError("every bin must contain at least one frame")
    if num_samples == 0:
        empty = torch.empty(0, dtype=torch.long, device=probabilities.device)
        return empty, empty

    sampled_bins = torch.multinomial(
        probabilities, num_samples=num_samples, replacement=True
    )
    sampled_starts = torch.floor(
        torch.rand(num_samples, device=probabilities.device)
        * (bin_ends[sampled_bins] - bin_starts[sampled_bins]).float()
    ).long() + bin_starts[sampled_bins]
    if pre_failure_window_steps > 0:
        offsets = torch.randint(
            pre_failure_window_steps,
            (num_samples,),
            device=probabilities.device,
        )
        sampled_starts = (sampled_starts - offsets).clamp_min(0)
    return bin_sequence_ids[sampled_bins], sampled_starts


def adaptive_phase_sampling_probabilities(
    failure_counts: torch.Tensor,
    visit_counts: torch.Tensor,
    *,
    bin_size_s: float,
    start_time_lo: float,
    start_time_hi: float,
    reference_duration: float,
    episode_duration: float,
    uniform_rate: float = 0.1,
    failure_rate_max_over_mean: float = 50.0,
    max_probability_multiplier: float = 50.0,
    valid_step_weighting: bool = True,
) -> torch.Tensor:
    """Build SONIC-style failure-adaptive probabilities over reference-time bins.

    Failure rates are mixed with a uniform-in-time floor. Bin overlap accounts
    for partial edge bins. ``valid_step_weighting`` additionally preserves the
    existing RobustTrack preference for starts with more non-padded targets.
    """
    if failure_counts.ndim != 1 or visit_counts.shape != failure_counts.shape:
        raise ValueError("failure_counts and visit_counts must be matching vectors")
    if not torch.is_floating_point(failure_counts) or not torch.is_floating_point(
        visit_counts
    ):
        raise ValueError("failure_counts and visit_counts must be floating point")
    if torch.any(failure_counts < 0.0) or torch.any(visit_counts <= 0.0):
        raise ValueError("counts must be nonnegative and visits must be positive")

    bin_size_s = float(bin_size_s)
    start_time_lo = float(start_time_lo)
    start_time_hi = float(start_time_hi)
    reference_duration = float(reference_duration)
    episode_duration = float(episode_duration)
    uniform_rate = float(uniform_rate)
    failure_rate_max_over_mean = float(failure_rate_max_over_mean)
    max_probability_multiplier = float(max_probability_multiplier)
    if bin_size_s <= 0.0:
        raise ValueError("bin_size_s must be positive")
    if not 0.0 <= uniform_rate <= 1.0:
        raise ValueError("uniform_rate must be in [0, 1]")
    if start_time_lo >= start_time_hi:
        raise ValueError("adaptive sampling requires start_time_lo < start_time_hi")
    if reference_duration <= 0.0 or episode_duration <= 0.0:
        raise ValueError("reference_duration and episode_duration must be positive")
    if failure_rate_max_over_mean <= 0.0:
        raise ValueError("failure_rate_max_over_mean must be positive")
    if max_probability_multiplier <= 0.0:
        raise ValueError("max_probability_multiplier must be positive")

    num_bins = failure_counts.numel()
    expected_bins = math.ceil(reference_duration / bin_size_s)
    if num_bins != expected_bins:
        raise ValueError(
            f"count vectors have {num_bins} bins, expected {expected_bins}"
        )

    dtype = torch.float64
    device = failure_counts.device
    bin_lo = torch.arange(num_bins, device=device, dtype=dtype) * bin_size_s
    bin_hi = torch.minimum(
        bin_lo + bin_size_s,
        torch.full_like(bin_lo, reference_duration),
    )
    support_lo = torch.clamp(bin_lo, min=start_time_lo, max=start_time_hi)
    support_hi = torch.clamp(bin_hi, min=start_time_lo, max=start_time_hi)
    support_width = (support_hi - support_lo).clamp(min=0.0)
    active = support_width > 0.0
    if not torch.any(active):
        raise ValueError("start-time range does not overlap any phase bin")

    base_weights = support_width
    if valid_step_weighting:
        midpoint = 0.5 * (support_lo + support_hi)
        valid_duration = torch.minimum(
            torch.full_like(midpoint, episode_duration),
            (reference_duration - midpoint).clamp(min=0.0),
        )
        base_weights *= valid_duration / episode_duration
    return adaptive_bin_sampling_probabilities(
        failure_counts,
        visit_counts,
        base_weights.to(dtype=failure_counts.dtype),
        uniform_rate=uniform_rate,
        failure_rate_max_over_mean=failure_rate_max_over_mean,
        max_probability_multiplier=max_probability_multiplier,
    )


def sample_adaptive_phase_start_times(
    probabilities: torch.Tensor,
    *,
    num_samples: int,
    bin_size_s: float,
    start_time_lo: float,
    start_time_hi: float,
    reference_duration: float,
    pre_failure_window_s: float = 0.0,
) -> torch.Tensor:
    """Sample uniformly inside adaptive bins, optionally starting before them."""
    if probabilities.ndim != 1 or probabilities.numel() < 1:
        raise ValueError("probabilities must be a non-empty vector")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if torch.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("probabilities must be nonnegative with positive sum")
    if start_time_lo > start_time_hi:
        raise ValueError("start_time_lo must not exceed start_time_hi")
    if pre_failure_window_s < 0.0:
        raise ValueError("pre_failure_window_s must be nonnegative")
    if num_samples == 0:
        return torch.empty(
            (0, 1), device=probabilities.device, dtype=probabilities.dtype
        )
    if start_time_lo == start_time_hi:
        return torch.full(
            (num_samples, 1),
            float(start_time_lo),
            device=probabilities.device,
            dtype=probabilities.dtype,
        )

    sampled_bins = torch.multinomial(
        probabilities, num_samples=num_samples, replacement=True
    )
    bin_lo = sampled_bins.to(probabilities.dtype) * float(bin_size_s)
    bin_hi = torch.minimum(
        bin_lo + float(bin_size_s),
        torch.full_like(bin_lo, float(reference_duration)),
    )
    sample_lo = bin_lo.clamp(min=float(start_time_lo), max=float(start_time_hi))
    sample_hi = bin_hi.clamp(min=float(start_time_lo), max=float(start_time_hi))
    starts = sample_lo + torch.rand_like(sample_lo) * (sample_hi - sample_lo)
    if pre_failure_window_s > 0.0:
        starts -= torch.rand_like(starts) * float(pre_failure_window_s)
    return starts.clamp(float(start_time_lo), float(start_time_hi)).unsqueeze(-1)


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
