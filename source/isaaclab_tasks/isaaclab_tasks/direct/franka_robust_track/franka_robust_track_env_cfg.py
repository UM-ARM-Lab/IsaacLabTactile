# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from omegaconf import OmegaConf

from isaaclab_tasks.direct.factory.factory_tasks_cfg import ASSET_DIR


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


@configclass
class CtrlCfg:
    """Low-level controller configuration."""

    backend: str = "factory_control"  # factory_control, dls_ik
    ema_factor: float = 1.0
    action_rep: str = "delta_ee_pose"
    # Controls when a delta action is converted to an absolute controller target:
    #   "per_physics_step" -> re-anchor it to the current EE pose on every
    #                         decimation substep (legacy behavior)
    #   "per_control_step" -> anchor it once in _pre_physics_step and hold the
    #                         resulting pose target throughout decimation
    # In both modes the policy still outputs the same normalized 6D delta action.
    delta_target_mode: str = "per_physics_step"

    # By default the OSC controller runs on the *nominal* mass matrix, rebuilt from
    # the default inertial parameters, so it stays blind to the payload and link-mass
    # randomization (that mismatch is the point of the robustness task). Set this to
    # True to instead feed the controller PhysX's ground-truth generalized mass matrix
    # (payload merge + link-mass scaling included), i.e. a perfectly-modeled controller.
    use_gt_mass_matrix: bool = False
    # Optional exact inertial model for the nominal OSC mass matrix. This is
    # separate from the PhysX plant so the controller can omit a physical tool
    # exactly as the NUC metadata model does.
    nominal_model_urdf: str | None = None
    apply_libfranka_torque_shaping: bool = False
    torque_command_limits = [86.0, 86.0, 86.0, 86.0, 11.5, 11.5, 11.5]
    actuator_hardware_hz: float = 1000.0
    actuator_filter_cutoff_hz: float = 100.0
    actuator_torque_rate_limit: float = 1000.0
    # Number of controller/physics ticks for which a newly published policy
    # target remains invisible. `delay_range` is sampled independently per env
    # and has inclusive bounds; equal bounds express a fixed delay. The old
    # scalar is retained only as a fallback when loading historical configs.
    target_update_delay_substeps: int = 0
    delay_range = None
    delay_mode: str = "per_episode"

    # Use the full operational-space control law: premultiply the task-space PD wrench
    # by the task inertia Λ = (J M⁻¹ Jᵀ)⁻¹ so the closed-loop task dynamics are unit
    # mass and the critical-damping gains (Kd = 2·√Kp) actually hold. Off gives the
    # legacy Jacobian-transpose PD, which is under-damped and oscillates in free space.
    use_task_space_inertia: bool = True
    singularity_robust_inverse: bool = False

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.15, 0.15, 0.15]

    reset_joints = [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
    default_task_prop_gains = [300.0, 300.0, 300.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Optional asymmetric reset-time multiplier range [low, high]. For example,
    # [0.5, 2.0] around a nominal gain of 80 samples gains uniformly in [40, 160].
    # When set, this replaces the symmetric `task_prop_gains_noise_level` scheme.
    task_prop_gains_randomization_scale_range = None
    # Reset-time gain randomization:
    #   "scalar"   -> draw one multiplier per env and scale all 6 nominal gains,
    #                 preserving their translation-to-rotation proportions
    #   "per_axis" -> draw an independent gain multiplier for each of the 6 axes
    # With no scale range, zero noise keeps the configured default gains.
    task_prop_gains_randomization_mode: str = "per_axis"
    # Optional reset-time damping-ratio randomization, applied after Kp is drawn.
    # One damping ratio is shared across all six axes in each environment to
    # match the uniform damping vector used by the hardware controller.
    task_deriv_gains_damping_ratio_range = None

    # If True, the policy also outputs the 6 task-space proportional gains
    # (3 translational, 3 rotational) as extra action dims, mapped from the
    # normalized [-1, 1] range onto [task_prop_gains_min, task_prop_gains_max].
    # The derivative gains are recomputed from the commanded proportional gains
    # each step. If False, gains stay at `default_task_prop_gains` (plus optional
    # reset-time noise). This grows the action space by 6.
    control_gains: bool = False
    # "per_axis" preserves the historical six gain actions; "scalar" uses one
    # action and applies the same scheduled Kp to all six task-space axes.
    control_gain_action_mode: str = "per_axis"
    control_gain_damping_ratio: float = 0.75
    task_prop_gains_min = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    task_prop_gains_max = [600.0, 600.0, 600.0, 60.0, 60.0, 60.0]
    joint_pos_kp: float = 80.0
    joint_pos_kd: float = 8.0

    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246
    use_full_rotation: bool = True

    # OSC nullspace posture target (the config the redundant DoF is biased toward):
    #   "start"          -> each env's own cuRobo start-pose IK solution (per-env; default)
    #   "reset_joints"   -> the fixed `reset_joints` config (collection-branch posture)
    #   "default_dof_pos"-> the fixed `default_dof_pos_tensor` (matches factory/forge)
    # Fixed anchors give a consistent arm posture instead of whatever IK branch each
    # sampled trajectory happens to land on.
    nullspace_posture: str = "start"


@configclass
class TrackingCfg:
    """Discretized end-effector reference trajectory configuration.

    The trajectory is generated analytically per env at reset and sampled onto
    a native reference grid. Synchronous policies consume one waypoint per
    action; async policies interpolate their active target between native knots.
    """

    mode: str = "line"  # line, line_fixed, circle, dataset

    # Straight-line trajectory: EE moves from its reset pose along a randomized
    # 3D direction, covering a randomized path length. Speed is not sampled
    # independently; the line is traversed exactly once over the full episode
    # (speed = length / episode_length), so the reference keeps moving the whole
    # episode regardless of the sampled length.
    line_length_range = [0.10, 0.30]  # m, total displacement along the line

    # "line_fixed" mode: identical to "line" but the direction is this fixed
    # vector (base frame, normalized internally) instead of a random one. Combined
    # with a fixed start pose (init.start_pos_box collapsed to a point,
    # start_rot_noise=0) and a collapsed line_length_range this yields the exact
    # same straight-line reference every reset.
    line_dir = [1.0, 0.0, 0.0]

    # Circle trajectory: EE traces a circle in a randomized plane. With the
    # default zero start-time offset, t=0 coincides with the sampled anchor pose.
    # Direction (cw/ccw) is randomized.
    circle_radius_range = [0.05, 0.15]  # m
    circle_speed_range = [0.20, 0.80]  # rad/s (angular speed)
    # Optional temporal phase offset for analytic circles. Zero preserves the
    # historical reset-at-the-start behavior.
    circle_start_time_range = [0.0, 0.0]  # s
    # Sampling distribution within circle_start_time_range. ``uniform`` keeps
    # the historical behavior. ``valid_steps`` weights each start by the
    # non-padded portion of one training episode, so starts near the finite
    # reference tail remain possible without over-representing terminal holds.
    circle_start_time_sampling: str = "uniform"  # uniform, valid_steps
    # Finite source-clip duration. References beyond this time hold the final
    # pose, matching the padded-tail behavior of fixed-length dataset clips.
    circle_reference_duration_s: float = 10.0
    # Optional trajectory-buffer capacity independent of the training episode.
    # Short-episode policies set this to the full reference duration so periodic
    # evaluation can run one continuous full trajectory without reallocating the
    # Isaac environment. Zero keeps the historical episode-sized allocation.
    reference_buffer_duration_s: float = 0.0

    # Optional orientation sweep: constant angular velocity about a randomized
    # axis, capped at a randomized total angle. Set ranges to 0 to hold fixed.
    rot_speed_range = [0.0, 0.20]  # rad/s
    rot_angle_range = [0.0, 0.30]  # rad, max sweep

    # Lookahead: the policy observes `num_future_steps` strictly future native
    # waypoints: [P_{r+1}, ..., P_{r+H}], where r is the native segment index.
    # The active reward target advances continuously from P_r to P_{r+1}.
    num_future_steps: int = 4
    # Physics substeps between native reference samples. Zero preserves the
    # synchronous contract by inheriting the policy/control decimation. Async
    # dataset tracking uses 8 here (15 Hz at 120 Hz physics) with a smaller
    # top-level decimation for a 30/60 Hz policy.
    reference_decimation: int = 0

    # "dataset" mode: instead of an analytic line/circle, each env tracks a real
    # end-effector trajectory sampled from an offline demonstration dataset (e.g.
    # the peg-insertion `tool_pose` sequences). No frame/representation conversion
    # is performed: the dataset pose must describe the frame selected by the env's
    # top-level `tool_frame`, as pos + quat (w, x, y, z) in the robot-base/env-local
    # frame, i.e. the exact format stored in `traj_pos_buf`/`traj_quat_buf`. At reset
    # a random episode/chunk is drawn. New training configs preserve raw per-step
    # samples and hold the selected chunk's last sample in the runtime tail; legacy
    # configs may still resample onto the per-step control grid. The usual
    # cuRobo reachability + singularity filter still applies (episodes that fall
    # outside the reachable/well-conditioned workspace are resampled/backfilled).
    dataset_path: str = ""  # HDF5 path; required when mode == "dataset"
    dataset_pose_key: str = "tool_pose"  # (N, 7) pos(3) + quat(4, wxyz) field
    # Forge critic state aligned with dataset_pose_key. Its first seven values
    # duplicate tool_pose and values [13:20] are the recorded Franka arm joints.
    # Short within-episode chunks use these joints for reset and history seeding.
    dataset_state_key: str = "low_dim_state"
    dataset_joint_pos_offset: int = 13
    dataset_only_success: bool = True  # keep only episodes with trial_success set
    dataset_max_trajs: int = 0  # cap loaded episodes (0 = all available)
    # "episode_start" reproduces full-trajectory training from raw sample zero;
    # "random" samples the reset timestep over the entire raw episode.
    dataset_chunk_start_mode: str = "episode_start"
    # Training chunk length in native reference poses. When positive, raw dataset
    # samples retain their original indices. The start mode selects sample zero or
    # a random timestep over the entire episode. Missing lead-in history repeats
    # the first sample; a chunk extending past the episode repeats its final sample.
    # Zero keeps the legacy resampling path for old configs/checkpoint replay.
    dataset_chunk_length: int = 0
    # Optional replay-only resampling length. A runtime timeout may be extended
    # to expose the final post-step state without changing the saved reference
    # grid. If this is shorter than the runtime trajectory buffer, the last
    # resampled pose is repeated to fill the extra lookahead slots.
    dataset_reference_length: int = 0  # 0 = use the full runtime trajectory buffer
    # Per-reset spatial augmentation for dataset trajectories. Available strategies:
    #   "smooth_bump": original quartic warp that preserves both endpoints and
    #                  reaches the sampled offset at the trajectory midpoint.
    #   "constant_scale_shape": every pose receives an offset of the same magnitude
    #                  while its direction rotates smoothly along a random cone.
    # For the shape strategy, `dataset_warp_shape_fraction` controls the cone's
    # transverse component (0 = rigid offset, 1 = pure loop), and
    # `dataset_warp_cycles_range` controls its number of turns. Zero probability
    # leaves the demonstration unchanged. Candidates still pass the cuRobo checks.
    dataset_warp_strategy: str = "smooth_bump"
    dataset_warp_prob: float = 0.0
    dataset_warp_pos_max: float = 0.0  # m
    dataset_warp_rot_max: float = 0.0  # rad
    dataset_warp_shape_fraction: float = 0.0
    dataset_warp_cycles_range = [0.5, 1.5]
    # When force tracking is on, source the per-step target wrench from this
    # dataset field so the applied disturbance matches the demonstration's real
    # contact wrench. The field may be (N, 3), in which case torque is padded
    # with zeros, or (N, 6) for complete force/torque training. Empty -> fall
    # back to the synthetic contact-band sampler. NOTE: in this peg dataset
    # `contact_force` is all zeros; the real interaction lives in `wrench`.
    dataset_force_key: str = "wrench"
    # Coordinate/sign convention of dataset_force_key. Forge records PhysX's
    # incoming-joint reaction in the force-sensor child-joint axes; virtual
    # contact instead generates an external force in world axes. Keep the legacy
    # world convention as the default for other datasets and opt Forge training
    # into ``sensor_child_joint_reaction`` explicitly in its launcher.
    dataset_wrench_convention: str = "world_external"
    # Remove the mean of the first N samples of each raw episode before using its
    # wrench as a force goal. This compensates the episode's static wrist-sensor
    # bias while preserving the demonstrated time-varying force profile. Zero
    # preserves the legacy raw-wrench behavior.
    dataset_force_bias_samples: int = 0
    # A dataset step counts as "contact" (for the free/contact metric split) when
    # its target-wrench magnitude exceeds this threshold (N).
    dataset_contact_force_threshold: float = 0.5

    # Force-tracking add-on (independent of `mode`). When enabled, a per-step
    # target-wrench sequence is sampled at reset alongside the pose trajectory and
    # fed to the policy (current + lookahead). ``force_mode`` determines whether
    # that goal is applied directly as a disturbance or tracked through virtual
    # contact and measured-force feedback. Synthetic analytic force trajectories
    # model *contact events*: the goal is zero
    # for most of the episode, punctuated by a few contact bands. Each band switches
    # on suddenly at a random onset and, while active, varies smoothly (continuous
    # magnitude oscillation plus a gentle direction drift) rather than staying
    # constant, then switches back off at the end of a sampled duration window.
    enable_force: bool = False
    # True uses force3+torque3 throughout observations, rewards, and application.
    # False preserves the legacy force-only contract and checkpoint shapes.
    use_full_wrench: bool = True
    # ``replay_disturbance`` applies the recorded six-axis target wrench directly
    # at the wrist. ``replay_raw_wrench`` loads all six dataset wrench
    # components, inverts their EMA assuming a zero initial wrench, and applies the
    # recovered force and torque. ``virtual_contact`` generates force from plane
    # penetration instead.
    force_mode: str = "replay_disturbance"
    dataset_wrench_ema_alpha: float = 0.25
    # Physical amplitude multipliers used only when replaying a dataset wrench as
    # an external disturbance. The target wrench exposed to the policy and used
    # by rewards remains in the original dataset units.
    disturbance_force_scale: float = 1.0
    disturbance_torque_scale: float = 1.0
    # Clip the norm of physically replayed dataset torque to this percentile of
    # the selected trajectories. This does not modify the dataset or the target
    # wrench exposed to the policy. 100 disables clipping.
    disturbance_torque_clip_percentile: float = 100.0
    force_mag_range = [5.0, 20.0]  # N, sampled peak contact magnitude (spec-capped at 20 N)
    torque_mag_max: float = 2.0  # Nm, observation normalization scale
    # Number of contact bands per episode, sampled per env (inclusive range). The
    # episode is split into that many equal segments, each holding one band, so the
    # bands never overlap and stay spread across the episode.
    force_num_bands_range = [2, 3]
    # Duration (s) each contact band stays active, sampled per band (capped so the
    # band fits inside its segment).
    force_duration_range = [0.3, 1.5]
    # Probability that a contact event occurs at all in an episode; with the
    # complementary probability the force stays zero for the whole episode.
    force_prob: float = 1.0
    # Frequency band (Hz) of the smooth oscillation of the force during contact.
    # The magnitude and the direction drift are built from a few sinusoids drawn
    # in this band, so the active force wobbles continuously instead of being flat.
    force_osc_freq_range = [0.5, 3.0]
    # Direction wobble amplitude (0 = fixed direction). The force direction drifts
    # smoothly around its sampled base direction by roughly this fraction.
    force_dir_drift: float = 0.3

    # Dataset-conditioned virtual contact. Every reference timestep gets its own
    # plane. The outward normal is a centered moving-average force direction, and
    # the plane is placed |F_goal| / k_p above the reference EE pose along that
    # normal. Consequently the reference pose produces the target force whenever
    # relative penetration rate is zero. k_p is fixed and independent of
    # policy-controlled robot stiffness.
    virtual_contact_goal_threshold: float = 1.0  # N
    virtual_contact_direction_smoothing_window: int = 9  # odd, centered, control samples
    # True linearly interpolates the dataset-conditioned surface point, surface
    # velocity, and target wrench at the physics rate, and interpolates then
    # renormalizes the surface normal. False zero-order holds all four fields at
    # their native reference samples (15 Hz for the async peg launcher).
    virtual_contact_interpolate_reference: bool = True
    virtual_contact_plane_stiffness: float = 1500.0  # k_p, N/m
    # Scale only the force used to place the virtual plane. The recorded wrench
    # remains the tracking/metric target. Values > 1 model a stronger physical
    # reaction than the measured dataset wrench at the demonstrated pose.
    virtual_contact_reference_force_scale: float = 1.0
    # Post-model multiplier on the virtual contact force physically applied to
    # the wrist. Virtual sensor feedback is divided by the same value before it
    # reaches the policy/reward, preserving the original dataset wrench units.
    # The plane geometry and dataset target are unchanged.
    virtual_contact_force_scale: float = 1.0
    # Independent post-model multiplier on physically applied contact torque.
    # Virtual torque feedback is divided by this value, independently of force.
    virtual_contact_torque_scale: float = 1.0
    # Independent scale for the demonstrated torque coupled to virtual contact.
    virtual_contact_reference_torque_scale: float = 1.0
    # Legacy replay scales demonstrated torque by normal-force magnitude.
    # The orientation models instead construct an offset equilibrium orientation
    # and compute torque from live orientation and angular-velocity error.
    virtual_contact_torque_model: Literal[
        "dataset_force_coupled", "orientation_spring", "orientation_power_law"
    ] = "dataset_force_coupled"
    virtual_contact_rotational_stiffness: float = 4.0  # Nm/rad
    virtual_contact_rotational_damping: float = 0.1  # Nm*s/rad
    # For orientation_power_law, K_p is derived from these reference values:
    # K_p = tau_ref / theta_ref**p.  The angular penetration is in radians.
    virtual_contact_rotational_power_exponent: float = 1.5
    virtual_contact_rotational_reference_torque: float = 0.2  # Nm
    virtual_contact_rotational_reference_penetration: float = 0.05  # rad
    # Select the normal-force law. The linear model uses a Kelvin-Voigt damping
    # coefficient derived from damping_ratio; Drake Hunt-Crossley and power_law
    # use the separate multiplicative dissipation coefficient below (units s/m).
    contact_model: Literal["linear", "drake_hunt_crossley", "power_law", "exponential"] = "linear"
    virtual_contact_damping_ratio: float = 0.7
    hunt_crossley_dissipation: float = 0.0  # s/m; zero disables multiplicative damping
    power_law_exponent: float = 1.5
    power_law_reference_force: float = 5.0  # N
    power_law_reference_penetration: float = 0.003  # m
    # Derived during __post_init__: f_ref / delta_ref**p, units N/m**p.
    power_law_coefficient: float = 5.0 / (0.003**1.5)
    exponential_sharpness: float = 2.0
    # Positive values clamp the Hunt-Crossley multiplier. Zero disables the clamp.
    max_damping_multiplier: float = 5.0
    virtual_contact_effective_mass: float = 1.0  # kg, used only to set damping
    # C1 smoothstep activation distance for the unilateral force. This ramps both
    # spring and damping contributions from zero instead of switching damping on
    # discontinuously at the first infinitesimal penetration.
    virtual_contact_transition_width: float = 0.0  # m; zero preserves exact F=k_p*penetration
    virtual_contact_force_cap: float = 20.0  # N; zero disables the clamp
    virtual_contact_torque_cap: float = 2.0  # Nm; zero disables the clamp

    # Reset-time, per-environment contact-domain randomization. Parameters are
    # sampled once and held for the complete episode so the actor can infer the
    # hidden contact law from its observation history. Disabled by default to
    # preserve existing checkpoint and replay behavior.
    virtual_contact_randomize_parameters: bool = False
    # Narrow ablation switch: randomize only the signed normal surface offset,
    # leaving the contact law and normal direction fixed.
    virtual_contact_randomize_surface_offset: bool = False
    power_law_exponent_range = [1.0, 1.6]
    power_law_reference_penetration_range = [0.003, 0.010]  # m, log-uniform
    hunt_crossley_dissipation_range = [0.0, 100.0]  # s/m
    virtual_contact_surface_offset_range = [-0.002, 0.002]  # m along tilted normal
    virtual_contact_normal_tilt_range = [0.0, 0.17453292519943295]  # rad, 0--10 deg
    virtual_contact_rotational_power_exponent_range = [1.1, 2.0]
    virtual_contact_rotational_reference_penetration_range = [0.2, 0.6]  # rad, log-uniform
    virtual_contact_rotational_damping_range = [0.0, 0.05]  # Nm*s/rad
    # TODO: evaluate physical-force-scale randomization as an isolated follow-up;
    # keep virtual_contact_force_scale fixed for the first adaptive-policy study.

    # Synthetic six-axis wrench sensor shown to the actor. Noise and reset-time
    # bias are disabled so training matches hierarchical Forge evaluation.
    # At the default 60 Hz physics rate, this has the same physical-time EMA
    # response as Forge's alpha=0.25 filter running at 120 Hz:
    # 1 - (1 - 0.25) ** (120 / 60) = 0.4375.
    force_sensor_smoothing_factor: float = 0.4375
    force_sensor_noise_std: float = 0.0  # N
    force_sensor_bias_range: float = 0.0  # N, per-axis reset-time bias
    torque_sensor_noise_std: float = 0.0  # Nm
    torque_sensor_bias_range: float = 0.0  # Nm, per-axis reset-time bias


@configclass
class InitCfg:
    """Randomized start pose and trajectory-reachability check at reset.

    At each reset a start EE pose is sampled (uniform position box around the
    nominal reset EE position, plus a random orientation perturbation). The
    trajectory is anchored at that start pose, and the start pose together with
    `reach_check_waypoints` evenly-spaced trajectory waypoints (spanning the
    episode) are solved with cuRobo IK; an env is kept only if every waypoint is
    reachable within the tolerances below, otherwise it is resampled. The robot
    is then placed at the cuRobo joint solution for the start pose.
    """

    # Start-position sampling (base/env-local frame, meters). If `start_pos_box`
    # is set to [[x_min, x_max], [y_min, y_max], [z_min, z_max]], the start EE
    # position is drawn uniformly from that absolute box. Otherwise it is drawn
    # from a box of half-extent `start_pos_noise` centered on the nominal reset EE.
    start_pos_box = None
    start_pos_noise = [0.10, 0.10, 0.10]  # m, per-axis half-extent of the start-pos box
    start_rot_noise = 0.5  # rad, max random rotation angle about a random axis

    reach_check_waypoints: int = 8  # trajectory waypoints checked for reachability (incl. start)
    reach_pos_tol: float = 0.002  # m, cuRobo IK position tolerance for success
    reach_rot_tol: float = 0.01  # rad, cuRobo IK orientation tolerance for success
    # Max allowed per-joint jump (rad) between consecutive waypoint IK solutions.
    # Each waypoint is solved seeded with the previous solution; an env is rejected
    # if any joint moves more than this between neighboring waypoints, so the whole
    # trajectory stays on a single continuous IK branch (no elbow flips / wrist
    # wraps) that the policy can actually track.
    reach_joint_diff_threshold: float = 0.5
    # Candidate trajectories drawn per env each resample attempt. The reachability
    # + continuity check rejects many samples, so oversampling evaluates this many
    # independent candidates per env in one batched IK pass and keeps the first
    # accepted one, drastically cutting the number of sequential resample attempts.
    reach_oversample: int = 4
    ik_num_seeds: int = 4  # cuRobo IK seeds per waypoint
    # Max IK queries solved per cuRobo call. A reset sends num_envs *
    # reach_check_waypoints queries; the worker splits larger requests into
    # chunks of this size to bound GPU memory (avoids OOM at high num_envs).
    ik_batch_size: int = 2048
    ik_robot_cfg: str = "franka.yml"  # cuRobo robot config (base=panda_link0, tool=panda_hand)
    # CUDA-graph replay in the cuRobo worker triggers an illegal memory access at
    # large batch sizes; keep it off (a few tenths of a second slower per solve).
    ik_use_cuda_graph: bool = False
    max_reach_attempts: int = 5  # resampling attempts before accepting the current sample
    # Seed the *first* trajectory-waypoint cuRobo IK solve with `ctrl.reset_joints`
    # (subsequent waypoints seed from the previous solution). This lands the whole
    # start config on the reset_joints arm branch instead of an arbitrary cuRobo seed,
    # matching how factory/forge teleport to reset_joints before their reset IK -- so a
    # tracked demonstration is followed on the same elbow/wrist branch it was collected on.
    seed_ik_with_reset_joints: bool = False

    # Uniform reset-state perturbations applied after a reachable IK start is
    # selected.  The reference trajectory is intentionally left unchanged.
    # A scalar is used for all seven arm joints; zero preserves the baseline.
    reset_joint_pos_noise: float = 0.0  # rad, half-width of q perturbation
    reset_joint_vel_noise: float = 0.0  # rad/s, half-width of qdot perturbation
    # Per-environment probability that both configured reset perturbations are
    # applied. One shared Bernoulli draw gates q and qdot so the remaining resets
    # stay exactly clean. A default of one preserves historical configurations.
    reset_noise_prob: float = 1.0

    # Optional whole-trajectory workspace gate. When enabled, every reference
    # position over the active episode must remain inside this axis-aligned box
    # in the robot base frame. This does not restrict the circle plane: vertical
    # motion is accepted as long as it remains within the configured bounds.
    enable_ee_containment: bool = False
    ee_containment_box = [[0.35, 0.60], [-0.25, 0.25], [0.35, 0.50]]

    # Singularity filter. On top of reachability + continuity, reject any candidate
    # trajectory whose per-waypoint arm configuration is too close to a kinematic
    # singularity. Near-singular configs are where the OSC task-space inertia
    # (J M⁻¹ Jᵀ)⁻¹ blows up, so small tracking errors explode into the rare
    # fully-diverged episodes that dominate the tracking-error tail. The metric is
    # computed from cuRobo's geometric Jacobian at the selected `tool_frame` (a
    # virtual frame is inserted into cuRobo's tree directly) from each IK solution.
    # A waypoint fails if
    # manipulability sqrt(det(J Jᵀ)) < `min_manipulability`
    # or condition number sigma_max/sigma_min > `max_jac_cond`.
    singularity_check: bool = True
    min_manipulability: float = 0.02
    max_jac_cond: float = 50.0


@configclass
class RandomizationCfg:
    """Dynamics randomization applied at reset."""

    enable_link_mass: bool = True
    link_mass_scale_range = [0.75, 1.25]

    enable_payload: bool = True
    # Uncompensated end-effector payload. The robot hand itself is already
    # represented in the model and compensated by the hardware controller.
    payload_mass_range = [0.0, 0.5]
    payload_com_range = [[-0.04, 0.04], [-0.04, 0.04], [-0.02, 0.12]]
    payload_body_name: str = "panda_hand"

    enable_joint_friction: bool = True
    # Per-joint static Coulomb friction bands, multiplied by one shared arm-wide
    # scale draw per environment. A single [lo, hi] pair remains supported for
    # compatibility and is broadcast to all seven joints.
    joint_friction_range = [
        [0.0, 5.0],
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.0, 0.5],
        [0.0, 0.5],
    ]
    joint_friction_scale_range = [1.0, 1.0]
    # Legacy nominal-times-scale scheme retained for old saved configurations.
    joint_friction_nominal = None
    joint_friction_jitter = 0.3

    # Per-joint reflected rotor inertia (kg·m²) added to the diagonal of the arm's
    # joint-space mass matrix in PhysX. Franka harmonic-drive joints have reflected
    # inertias on the order of ~0.1 kg·m² (the real OSC controller lumps ~0.15 onto
    # the wrist joints), so this range brackets the physical value with margin. The
    # OSC controller stays blind to it — its nominal mass matrix keeps the
    # config-default armature — so the sampled armature is a modeled-vs-real inertia
    # mismatch the policy must be robust to.
    enable_joint_armature: bool = True
    joint_armature_range = [[0.0, 0.3]] * 7
    joint_armature_scale_range = [1.0, 1.0]

    # Sliding Coulomb friction as a fraction of static friction. PhysX requires
    # the dynamic coefficient to be no larger than the static coefficient.
    enable_joint_dynamic_ratio: bool = True
    joint_dynamic_ratio_range = [[1.0, 1.0]] * 7
    joint_dynamic_ratio_scale_range = [1.0, 1.0]

    # Solver-level viscous friction, proportional to joint velocity.
    enable_joint_viscous: bool = True
    joint_viscous_range = [[0.0, 0.0]] * 7
    joint_viscous_scale_range = [1.0, 1.0]

    # Integer command dead time in policy/control steps, sampled per environment
    # at reset. The bounds are inclusive.
    enable_action_latency: bool = False
    action_latency_range = [0, 1]


@configclass
class RewardCfg:
    """Reward scales for robust Cartesian tracking."""

    pos_error_scale: float = 8.0
    rot_error_scale: float = 2.0
    # Exponential kernel temperatures: reward = exp(-error / temp). These must be
    # matched to the error magnitude the policy actually operates at, otherwise the
    # kernel saturates to ~0 and provides no gradient (rot error runs ~1 rad, so a
    # tight temp like 0.1 leaves it in a flat dead zone).
    pos_error_temp: float = 0.01  # m
    rot_error_temp: float = 0.3  # rad
    # Narrow kernels refine tracking once the broad terms have brought the
    # policy close to the reference. Rotation uses a broad 0.3-rad recovery
    # kernel above and an equally weighted 0.1-rad refinement kernel below.
    fine_pos_error_scale: float = 0.0
    fine_rot_error_scale: float = 2.0
    fine_pos_error_temp: float = 0.03  # m
    fine_rot_error_temp: float = 0.1  # rad
    ee_vel_scale: float = -0.01
    # Physical finite-difference derivatives at the policy timestep. The caps
    # prevent reset/contact transients from dominating the tracking objective.
    ee_derivative_mode: str = "physical"  # physical, normalized_difference
    ee_accel_scale: float = -10.0
    ee_accel_clip: float = 0.025  # <= 0 disables clipping
    ee_jerk_scale: float = -10.0
    ee_jerk_clip: float = 0.1  # <= 0 disables clipping
    action_rate_scale: float = -0.02
    # Penalize consecutive action increments that reverse sign. Unlike action
    # rate, this directly targets Nyquist-like bang-bang alternation.
    action_alternation_scale: float = 0.0
    # Penalize step-to-step change in the commanded controller gains (only active
    # when ctrl.control_gains is True) to encourage smooth gain scheduling instead
    # of chattering stiffness. Computed on the normalized [-1, 1] gain actions.
    gain_rate_scale: float = -0.02
    # Virtual-contact wrench regulation. Force and torque vector errors use
    # separate scales/temperatures because their units differ.
    force_error_scale: float = 4.0
    force_error_temp: float = 1.0  # N
    torque_error_scale: float = 4.0
    torque_error_temp: float = 0.1  # Nm
    force_over_scale: float = -1.0
    force_over_margin: float = 0.5  # N
    force_safe_scale: float = -2.0
    force_safe_threshold: float = 10.0  # N
    force_success_threshold: float = 1.0  # N absolute normal-force error
    torque_success_threshold: float = 0.1  # Nm vector torque error
    # Weak privileged shaping of the normal gain: high far from the plane and low
    # near/in contact. The actor does not observe exact plane distance, so it must
    # infer the schedule from trajectory/force-goal lookahead and force feedback.
    stiffness_preference_scale: float = -0.5
    stiffness_near_margin: float = 0.005  # m
    stiffness_near_temperature: float = 0.002  # m
    # Relax normal pose tracking near contact so force regulation can move the EE
    # slightly away from the demonstrated normal position; tangential tracking is
    # unchanged.
    contact_normal_pose_weight: float = 0.2
    joint_vel_scale: float = -0.002
    joint_limit_scale: float = -0.05
    success_pos_threshold: float = 0.001
    success_rot_threshold: float = 0.05


@configclass
class FrankaRobustTrackEnvCfg(DirectRLEnvCfg):
    """Robot-only Franka environment for robust commanded EE tracking."""

    episode_length_s = 10.0
    decimation = 4
    action_space = 6

    # Optional observation history: if > 1, the policy and critic observations are
    # the last `obs_history_length` single-step frames concatenated oldest->newest
    # into one flat vector (frame history replaces the recurrent memory). H = 1
    # reproduces the original single-step observation.
    obs_history_length: int = 1
    # Include the seven arm joint angles in each proprioceptive observation
    # frame. Disable this when training a tracker that should rely only on the
    # fingertip pose and its previous action history.
    include_joint_angles: bool = True
    # Include the seven arm joint velocities in each proprioceptive history
    # frame. Disabled by default to preserve the observation contract of
    # existing policies.
    include_joint_velocities: bool = False
    # Append the six current task-space proportional gains to the policy and
    # critic observations. Disable this when gains are fixed and the policy
    # should use the legacy observation contract.
    controller_context_in_policy: bool = True
    # Give the asymmetric critic the sampled controller state without exposing
    # it to the actor. When enabled, the critic receives both the six
    # proportional and six derivative task-space gains. If the policy already
    # receives Kp, only Kd is appended again on the critic-only path.
    controller_context_in_critic: bool = False
    # Debug guard that synchronizes policy/critic tensors back to the CPU each
    # step. Offline replay disables it after startup validation for throughput.
    validate_observations: bool = True

    # Debug visualization of the reference trajectory: current command frame,
    # lookahead targets, and the full episode path (sampled in time).
    debug_vis: bool = False
    debug_vis_path_samples: int = 40
    # Virtual-contact force visualization. Component mode renders signed Fx/Fy/Fz
    # as red/green/blue arrows along the world axes. Vector mode renders their
    # resultant. Yellow is used for the plane outward-normal arrow, and the
    # translucent cyan patch is the active virtual plane.
    debug_vis_force: bool = False
    debug_vis_force_style: Literal["components", "vector", "both"] = "components"
    debug_vis_force_source: Literal["reference", "realized", "both"] = "reference"
    debug_vis_force_scale: float = 0.02  # rendered arrow length, m/N
    debug_vis_force_shaft_radius: float = 0.0025  # cylindrical shaft radius, m
    debug_vis_force_head_radius: float = 0.008  # conical arrowhead radius, m
    debug_vis_force_head_length: float = 0.018  # maximum arrowhead length, m
    debug_vis_force_plane_size: float = 0.08  # square plane side length, m
    debug_vis_force_normal_length: float = 0.06  # rendered unit-normal arrow, m
    debug_vis_force_ee_sphere: bool = True
    debug_vis_force_ee_sphere_radius: float = 0.002  # current EE marker radius, m

    # Per-episode-step tracking-error profile logged to wandb as a mean±std line
    # plot. The pos/rot tracking error is binned by the step index within the
    # episode and accumulated over this many episodes, after which the profile is
    # logged and the accumulators are reset. Set <= 0 to disable the logging.
    log_perstep_error_episodes: int = 5
    # observation_space / state_space are recomputed in __post_init__. Only the
    # proprioceptive part (20 dims by default, 27 with joint velocities, or 13
    # when both joint fields are disabled) is stacked over obs_history_length
    # frames. The lookahead future errors (6 * num_future_steps), controller
    # gains (6), and the critic's privileged dims are episode-constant/current
    # context, so they are appended once rather than duplicated across history.
    observation_space = 50
    state_space = 74

    ctrl: CtrlCfg = CtrlCfg()
    tracking: TrackingCfg = TrackingCfg()
    init: InitCfg = InitCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    reward: RewardCfg = RewardCfg()

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 60,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # gpu_max_rigid_contact_count=2**23,
            # gpu_max_rigid_patch_count=2**23,
            # gpu_collision_stack_size=2**29,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    viewer: ViewerCfg = ViewerCfg(
        # Offsets are relative to the robot base (origin_type="asset_root").
        # 3/4 elevated view pulled back to frame the full arm and the tracking
        # workspace (base-frame center ~ (0.45, 0.0, 0.3)).
        eye=(1.6, 1.1, 0.8),
        lookat=(0.45, 0.45, 0.45),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(720, 720),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_gelsight_mini_assembled_z13_x10.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    # New RobustTrack training uses the same Gelsight Franka as Forge. Saved
    # mimic checkpoints retain their serialized asset config, and launchers can
    # still request ``franka_mimic.usd`` explicitly for reproduction.
    robot_usd_path: str = "franka_gelsight_mini_assembled_z13_x10.usd"
    # Optional inertial override for the PhysX plant. USD-only bodies absent
    # from this model are reduced to numerically valid negligible placeholders.
    plant_model_urdf: str | None = None
    # Frame whose pose/Jacobian defines the controlled and observed EE. "auto"
    # preserves the legacy rigid-body priority: panda_fingertip_centered,
    # force_sensor, then panda_hand. "fr3_wsg_tcp" is a virtual frame computed from
    # panda_hand using the fixed transform in panda_arm_wsg_horizontal.urdf; it does
    # not need to exist in the USD.
    tool_frame: str = "auto"
    params = None

    def update_env_params(self):
        if self.params is None:
            self.params = OmegaConf.create({})
        env = self.params.get("env", OmegaConf.create({}))

        if env.get("robot_usd_path", None) is not None:
            self.robot_usd_path = env.robot_usd_path
        if env.get("plant_model_urdf", None) is not None:
            self.plant_model_urdf = str(env.plant_model_urdf)
        if env.get("tool_frame", None) is not None:
            self.tool_frame = str(env.tool_frame)
        if env.get("obs_history_length", None) is not None:
            self.obs_history_length = int(env.obs_history_length)
        if env.get("include_joint_angles", None) is not None:
            self.include_joint_angles = bool(env.include_joint_angles)
        if env.get("include_joint_velocities", None) is not None:
            self.include_joint_velocities = bool(env.include_joint_velocities)
        if env.get("controller_context_in_policy", None) is not None:
            self.controller_context_in_policy = bool(env.controller_context_in_policy)
        if env.get("controller_context_in_critic", None) is not None:
            self.controller_context_in_critic = bool(env.controller_context_in_critic)
        if env.get("debug_vis", None) is not None:
            self.debug_vis = env.debug_vis
        if env.get("debug_vis_path_samples", None) is not None:
            self.debug_vis_path_samples = env.debug_vis_path_samples
        if env.get("debug_vis_force", None) is not None:
            self.debug_vis_force = bool(env.debug_vis_force)
        if env.get("debug_vis_force_style", None) is not None:
            self.debug_vis_force_style = str(env.debug_vis_force_style)
        if env.get("debug_vis_force_source", None) is not None:
            self.debug_vis_force_source = str(env.debug_vis_force_source)
        if env.get("debug_vis_force_scale", None) is not None:
            self.debug_vis_force_scale = float(env.debug_vis_force_scale)
        if env.get("debug_vis_force_shaft_radius", None) is not None:
            self.debug_vis_force_shaft_radius = float(env.debug_vis_force_shaft_radius)
        if env.get("debug_vis_force_head_radius", None) is not None:
            self.debug_vis_force_head_radius = float(env.debug_vis_force_head_radius)
        if env.get("debug_vis_force_head_length", None) is not None:
            self.debug_vis_force_head_length = float(env.debug_vis_force_head_length)
        if env.get("debug_vis_force_plane_size", None) is not None:
            self.debug_vis_force_plane_size = float(env.debug_vis_force_plane_size)
        if env.get("debug_vis_force_normal_length", None) is not None:
            self.debug_vis_force_normal_length = float(env.debug_vis_force_normal_length)
        if env.get("debug_vis_force_ee_sphere", None) is not None:
            self.debug_vis_force_ee_sphere = bool(env.debug_vis_force_ee_sphere)
        if env.get("debug_vis_force_ee_sphere_radius", None) is not None:
            self.debug_vis_force_ee_sphere_radius = float(
                env.debug_vis_force_ee_sphere_radius
            )
        if env.get("log_perstep_error_episodes", None) is not None:
            self.log_perstep_error_episodes = int(env.log_perstep_error_episodes)
        # Physics-step control: `decimation` sim substeps per control step and the
        # physics `sim.dt`. step_dt = decimation * sim.dt sets the control rate, so to
        # match the factory/forge peg-collection env exactly use dt=1/120 + decimation=8
        # (120 Hz physics, 15 Hz control) instead of the default 1/60 + 4.
        if env.get("episode_length_s", None) is not None:
            self.episode_length_s = float(env.episode_length_s)
        if env.get("decimation", None) is not None:
            self.decimation = int(env.decimation)
        sim = env.get("sim", OmegaConf.create({}))
        if sim.get("dt", None) is not None:
            self.sim.dt = float(sim.dt)

        ctrl = env.get("ctrl", OmegaConf.create({}))
        for key in [
            "backend",
            "ema_factor",
            "action_rep",
            "delta_target_mode",
            "use_gt_mass_matrix",
            "nominal_model_urdf",
            "apply_libfranka_torque_shaping",
            "torque_command_limits",
            "actuator_hardware_hz",
            "actuator_filter_cutoff_hz",
            "actuator_torque_rate_limit",
            "target_update_delay_substeps",
            "delay_range",
            "delay_mode",
            "use_task_space_inertia",
            "singularity_robust_inverse",
            "pos_action_threshold",
            "rot_action_threshold",
            "default_task_prop_gains",
            "task_prop_gains_noise_level",
            "task_prop_gains_randomization_scale_range",
            "task_prop_gains_randomization_mode",
            "task_deriv_gains_damping_ratio_range",
            "control_gains",
            "control_gain_action_mode",
            "control_gain_damping_ratio",
            "task_prop_gains_min",
            "task_prop_gains_max",
            "reset_joints",
            "joint_pos_kp",
            "joint_pos_kd",
            "nullspace_posture",
        ]:
            if ctrl.get(key, None) is not None:
                setattr(self.ctrl, key, _to_plain(ctrl[key]))

        tracking = env.get("tracking", OmegaConf.create({}))
        for key in [
            "mode",
            "line_length_range",
            "line_dir",
            "circle_radius_range",
            "circle_speed_range",
            "circle_start_time_range",
            "circle_start_time_sampling",
            "circle_reference_duration_s",
            "reference_buffer_duration_s",
            "rot_speed_range",
            "rot_angle_range",
            "num_future_steps",
            "reference_decimation",
            "dataset_path",
            "dataset_pose_key",
            "dataset_state_key",
            "dataset_joint_pos_offset",
            "dataset_only_success",
            "dataset_max_trajs",
            "dataset_chunk_start_mode",
            "dataset_chunk_length",
            "dataset_reference_length",
            "dataset_warp_strategy",
            "dataset_warp_prob",
            "dataset_warp_pos_max",
            "dataset_warp_rot_max",
            "dataset_warp_shape_fraction",
            "dataset_warp_cycles_range",
            "dataset_force_key",
            "dataset_wrench_convention",
            "dataset_force_bias_samples",
            "dataset_contact_force_threshold",
            "enable_force",
            "use_full_wrench",
            "force_mode",
            "dataset_wrench_ema_alpha",
            "disturbance_force_scale",
            "disturbance_torque_scale",
            "disturbance_torque_clip_percentile",
            "force_mag_range",
            "torque_mag_max",
            "force_num_bands_range",
            "force_duration_range",
            "force_prob",
            "force_osc_freq_range",
            "force_dir_drift",
            "virtual_contact_goal_threshold",
            "virtual_contact_direction_smoothing_window",
            "virtual_contact_interpolate_reference",
            "virtual_contact_plane_stiffness",
            "virtual_contact_reference_force_scale",
            "virtual_contact_force_scale",
            "virtual_contact_torque_scale",
            "virtual_contact_reference_torque_scale",
            "virtual_contact_torque_model",
            "virtual_contact_rotational_stiffness",
            "virtual_contact_rotational_damping",
            "virtual_contact_rotational_power_exponent",
            "virtual_contact_rotational_reference_torque",
            "virtual_contact_rotational_reference_penetration",
            "contact_model",
            "virtual_contact_damping_ratio",
            "hunt_crossley_dissipation",
            "power_law_exponent",
            "power_law_reference_force",
            "power_law_reference_penetration",
            "power_law_coefficient",
            "exponential_sharpness",
            "max_damping_multiplier",
            "virtual_contact_effective_mass",
            "virtual_contact_transition_width",
            "virtual_contact_force_cap",
            "virtual_contact_torque_cap",
            "virtual_contact_randomize_parameters",
            "virtual_contact_randomize_surface_offset",
            "power_law_exponent_range",
            "power_law_reference_penetration_range",
            "hunt_crossley_dissipation_range",
            "virtual_contact_surface_offset_range",
            "virtual_contact_normal_tilt_range",
            "virtual_contact_rotational_power_exponent_range",
            "virtual_contact_rotational_reference_penetration_range",
            "virtual_contact_rotational_damping_range",
            "force_sensor_smoothing_factor",
            "force_sensor_noise_std",
            "force_sensor_bias_range",
            "torque_sensor_noise_std",
            "torque_sensor_bias_range",
        ]:
            if tracking.get(key, None) is not None:
                setattr(self.tracking, key, _to_plain(tracking[key]))

        init = env.get("init", OmegaConf.create({}))
        for key in [
            "start_pos_box",
            "start_pos_noise",
            "start_rot_noise",
            "reach_check_waypoints",
            "reach_pos_tol",
            "reach_rot_tol",
            "reach_joint_diff_threshold",
            "reach_oversample",
            "ik_num_seeds",
            "ik_batch_size",
            "ik_robot_cfg",
            "ik_use_cuda_graph",
            "max_reach_attempts",
            "seed_ik_with_reset_joints",
            "reset_joint_pos_noise",
            "reset_joint_vel_noise",
            "reset_noise_prob",
            "enable_ee_containment",
            "ee_containment_box",
            "singularity_check",
            "min_manipulability",
            "max_jac_cond",
        ]:
            if init.get(key, None) is not None:
                setattr(self.init, key, _to_plain(init[key]))

        randomization = env.get("randomization", OmegaConf.create({}))
        for key in [
            "enable_link_mass",
            "link_mass_scale_range",
            "enable_payload",
            "payload_mass_range",
            "payload_com_range",
            "payload_body_name",
            "enable_joint_friction",
            "joint_friction_range",
            "joint_friction_nominal",
            "joint_friction_scale_range",
            "joint_friction_jitter",
            "enable_joint_armature",
            "joint_armature_range",
            "joint_armature_scale_range",
            "enable_joint_dynamic_ratio",
            "joint_dynamic_ratio_range",
            "joint_dynamic_ratio_scale_range",
            "enable_joint_viscous",
            "joint_viscous_range",
            "joint_viscous_scale_range",
            "enable_action_latency",
            "action_latency_range",
        ]:
            if randomization.get(key, None) is not None:
                value = randomization[key]
                setattr(self.randomization, key, _to_plain(value))

        reward = env.get("reward", OmegaConf.create({}))
        for key in [
            "pos_error_scale",
            "rot_error_scale",
            "pos_error_temp",
            "rot_error_temp",
            "fine_pos_error_scale",
            "fine_rot_error_scale",
            "fine_pos_error_temp",
            "fine_rot_error_temp",
            "ee_vel_scale",
            "ee_derivative_mode",
            "ee_accel_scale",
            "ee_accel_clip",
            "ee_jerk_scale",
            "ee_jerk_clip",
            "action_rate_scale",
            "action_alternation_scale",
            "joint_vel_scale",
            "joint_limit_scale",
            "gain_rate_scale",
            "force_error_scale",
            "force_error_temp",
            "torque_error_scale",
            "torque_error_temp",
            "force_over_scale",
            "force_over_margin",
            "force_safe_scale",
            "force_safe_threshold",
            "force_success_threshold",
            "torque_success_threshold",
            "stiffness_preference_scale",
            "stiffness_near_margin",
            "stiffness_near_temperature",
            "contact_normal_pose_weight",
            "success_pos_threshold",
            "success_rot_threshold",
        ]:
            if reward.get(key, None) is not None:
                setattr(self.reward, key, _to_plain(reward[key]))

    def __post_init__(self):
        self.update_env_params()
        for name in (
            "pos_error_temp",
            "rot_error_temp",
            "fine_pos_error_temp",
            "fine_rot_error_temp",
        ):
            if float(getattr(self.reward, name)) <= 0.0:
                raise ValueError(f"reward.{name} must be positive")
        if self.reward.ee_derivative_mode not in ("physical", "normalized_difference"):
            raise ValueError(
                "reward.ee_derivative_mode must be 'physical' or "
                f"'normalized_difference', got {self.reward.ee_derivative_mode!r}"
            )
        for name in ("ee_accel_clip", "ee_jerk_clip"):
            if not math.isfinite(float(getattr(self.reward, name))):
                raise ValueError(f"reward.{name} must be finite")
        if self.debug_vis_force_style not in ("components", "vector", "both"):
            raise ValueError(
                "debug_vis_force_style must be 'components', 'vector', or 'both'"
            )
        if self.debug_vis_force_source not in ("reference", "realized", "both"):
            raise ValueError(
                "debug_vis_force_source must be 'reference', 'realized', or 'both'"
            )
        if float(self.debug_vis_force_scale) <= 0.0:
            raise ValueError("debug_vis_force_scale must be positive")
        if float(self.debug_vis_force_shaft_radius) <= 0.0:
            raise ValueError("debug_vis_force_shaft_radius must be positive")
        if float(self.debug_vis_force_head_radius) <= 0.0:
            raise ValueError("debug_vis_force_head_radius must be positive")
        if float(self.debug_vis_force_head_length) <= 0.0:
            raise ValueError("debug_vis_force_head_length must be positive")
        if float(self.debug_vis_force_plane_size) <= 0.0:
            raise ValueError("debug_vis_force_plane_size must be positive")
        if float(self.debug_vis_force_normal_length) <= 0.0:
            raise ValueError("debug_vis_force_normal_length must be positive")
        if float(self.debug_vis_force_ee_sphere_radius) <= 0.0:
            raise ValueError("debug_vis_force_ee_sphere_radius must be positive")
        self.tracking.num_future_steps = int(self.tracking.num_future_steps)
        self.tracking.circle_start_time_sampling = str(
            self.tracking.circle_start_time_sampling
        )
        if self.tracking.circle_start_time_sampling not in ("uniform", "valid_steps"):
            raise ValueError(
                "tracking.circle_start_time_sampling must be 'uniform' or "
                f"'valid_steps', got {self.tracking.circle_start_time_sampling!r}"
            )
        self.tracking.reference_decimation = int(
            self.tracking.reference_decimation
        )
        if self.tracking.reference_decimation == 0:
            self.tracking.reference_decimation = self.decimation
        if self.tracking.reference_decimation < self.decimation:
            raise ValueError(
                "tracking.reference_decimation must be at least env.decimation"
            )
        if self.tracking.reference_decimation % self.decimation != 0:
            raise ValueError(
                "tracking.reference_decimation must be an integer multiple of "
                "env.decimation"
            )
        self.tracking.force_mode = str(self.tracking.force_mode)
        if self.tracking.force_mode not in (
            "replay_disturbance",
            "replay_raw_wrench",
            "virtual_contact",
        ):
            raise ValueError(
                "tracking.force_mode must be 'replay_disturbance', 'replay_raw_wrench', "
                "or 'virtual_contact', "
                f"got {self.tracking.force_mode!r}"
            )
        self.tracking.dataset_wrench_convention = str(
            self.tracking.dataset_wrench_convention
        )
        if self.tracking.dataset_wrench_convention not in (
            "world_external",
            "sensor_child_joint_reaction",
        ):
            raise ValueError(
                "tracking.dataset_wrench_convention must be 'world_external' or "
                "'sensor_child_joint_reaction', got "
                f"{self.tracking.dataset_wrench_convention!r}"
            )
        if not 0.0 < float(self.tracking.dataset_wrench_ema_alpha) <= 1.0:
            raise ValueError("tracking.dataset_wrench_ema_alpha must be in (0, 1]")
        if float(self.tracking.disturbance_force_scale) < 0.0:
            raise ValueError("tracking.disturbance_force_scale must be non-negative")
        if float(self.tracking.disturbance_torque_scale) < 0.0:
            raise ValueError("tracking.disturbance_torque_scale must be non-negative")
        torque_clip_percentile = float(
            self.tracking.disturbance_torque_clip_percentile
        )
        if not 0.0 < torque_clip_percentile <= 100.0:
            raise ValueError(
                "tracking.disturbance_torque_clip_percentile must be in (0, 100]"
            )
        self.tracking.use_full_wrench = bool(self.tracking.use_full_wrench)
        if self.tracking.force_mode == "replay_raw_wrench" and not self.tracking.use_full_wrench:
            raise ValueError(
                "tracking.force_mode='replay_raw_wrench' requires use_full_wrench=True"
            )
        if float(self.tracking.torque_mag_max) <= 0.0:
            raise ValueError("tracking.torque_mag_max must be positive")
        self.tracking.dataset_force_bias_samples = int(self.tracking.dataset_force_bias_samples)
        if self.tracking.dataset_force_bias_samples < 0:
            raise ValueError("tracking.dataset_force_bias_samples must be non-negative")
        if self.tracking.enable_force and self.tracking.force_mode == "virtual_contact":
            direction_window = int(self.tracking.virtual_contact_direction_smoothing_window)
            self.tracking.virtual_contact_direction_smoothing_window = direction_window
            if direction_window < 1 or direction_window % 2 == 0:
                raise ValueError(
                    "tracking.virtual_contact_direction_smoothing_window must be a positive odd integer"
                )
            if float(self.tracking.virtual_contact_plane_stiffness) <= 0.0:
                raise ValueError("tracking.virtual_contact_plane_stiffness must be positive")
            if float(self.tracking.virtual_contact_reference_force_scale) <= 0.0:
                raise ValueError("tracking.virtual_contact_reference_force_scale must be positive")
            if float(self.tracking.virtual_contact_force_scale) <= 0.0:
                raise ValueError("tracking.virtual_contact_force_scale must be positive")
            if float(self.tracking.virtual_contact_torque_scale) <= 0.0:
                raise ValueError("tracking.virtual_contact_torque_scale must be positive")
            if float(self.tracking.virtual_contact_reference_torque_scale) <= 0.0:
                raise ValueError("tracking.virtual_contact_reference_torque_scale must be positive")
            self.tracking.virtual_contact_torque_model = str(
                self.tracking.virtual_contact_torque_model
            )
            if self.tracking.virtual_contact_torque_model not in (
                "dataset_force_coupled",
                "orientation_spring",
                "orientation_power_law",
            ):
                raise ValueError(
                    "tracking.virtual_contact_torque_model must be "
                    "'dataset_force_coupled', 'orientation_spring', or "
                    "'orientation_power_law'"
                )
            if float(self.tracking.virtual_contact_rotational_stiffness) <= 0.0:
                raise ValueError(
                    "tracking.virtual_contact_rotational_stiffness must be positive"
                )
            if float(self.tracking.virtual_contact_rotational_damping) < 0.0:
                raise ValueError(
                    "tracking.virtual_contact_rotational_damping must be non-negative"
                )
            if float(self.tracking.virtual_contact_rotational_power_exponent) <= 0.0:
                raise ValueError(
                    "tracking.virtual_contact_rotational_power_exponent must be positive"
                )
            if float(self.tracking.virtual_contact_rotational_reference_torque) <= 0.0:
                raise ValueError(
                    "tracking.virtual_contact_rotational_reference_torque must be positive"
                )
            if float(self.tracking.virtual_contact_rotational_reference_penetration) <= 0.0:
                raise ValueError(
                    "tracking.virtual_contact_rotational_reference_penetration must be positive"
                )
            self.tracking.contact_model = str(self.tracking.contact_model)
            if self.tracking.contact_model not in (
                "linear",
                "drake_hunt_crossley",
                "power_law",
                "exponential",
            ):
                raise ValueError(
                    "tracking.contact_model must be 'linear', 'drake_hunt_crossley', "
                    "'power_law', or 'exponential', "
                    f"got {self.tracking.contact_model!r}"
                )
            if float(self.tracking.virtual_contact_damping_ratio) < 0.0:
                raise ValueError("tracking.virtual_contact_damping_ratio must be non-negative")
            if float(self.tracking.hunt_crossley_dissipation) < 0.0:
                raise ValueError("tracking.hunt_crossley_dissipation must be non-negative")
            exponent = float(self.tracking.power_law_exponent)
            reference_force = float(self.tracking.power_law_reference_force)
            reference_penetration = float(self.tracking.power_law_reference_penetration)
            if exponent <= 0.0:
                raise ValueError("tracking.power_law_exponent must be positive")
            if reference_force <= 0.0:
                raise ValueError("tracking.power_law_reference_force must be positive")
            if reference_penetration <= 0.0:
                raise ValueError("tracking.power_law_reference_penetration must be positive")
            self.tracking.power_law_coefficient = reference_force / reference_penetration**exponent
            if float(self.tracking.exponential_sharpness) <= 0.0:
                raise ValueError("tracking.exponential_sharpness must be positive")
            if float(self.tracking.max_damping_multiplier) < 0.0:
                raise ValueError("tracking.max_damping_multiplier must be non-negative")
            if float(self.tracking.virtual_contact_force_cap) < 0.0:
                raise ValueError("tracking.virtual_contact_force_cap must be non-negative")
            if float(self.tracking.virtual_contact_torque_cap) < 0.0:
                raise ValueError("tracking.virtual_contact_torque_cap must be non-negative")
            if float(self.tracking.virtual_contact_transition_width) < 0.0:
                raise ValueError("tracking.virtual_contact_transition_width must be non-negative")
            self.tracking.virtual_contact_randomize_parameters = bool(
                self.tracking.virtual_contact_randomize_parameters
            )
            self.tracking.virtual_contact_randomize_surface_offset = bool(
                self.tracking.virtual_contact_randomize_surface_offset
            )
            randomization_ranges = {
                "power_law_exponent_range": (True, False),
                "power_law_reference_penetration_range": (True, True),
                "hunt_crossley_dissipation_range": (False, False),
                "virtual_contact_surface_offset_range": (False, False),
                "virtual_contact_normal_tilt_range": (False, False),
                "virtual_contact_rotational_power_exponent_range": (True, False),
                "virtual_contact_rotational_reference_penetration_range": (True, True),
                "virtual_contact_rotational_damping_range": (False, False),
            }
            for name, (positive, log_uniform) in randomization_ranges.items():
                values = list(getattr(self.tracking, name))
                if len(values) != 2:
                    raise ValueError(f"tracking.{name} must be [low, high]")
                low, high = (float(value) for value in values)
                if low > high:
                    raise ValueError(f"tracking.{name} must satisfy low <= high")
                if positive and low <= 0.0:
                    raise ValueError(f"tracking.{name} lower bound must be positive")
                if not positive and name != "virtual_contact_surface_offset_range" and low < 0.0:
                    raise ValueError(f"tracking.{name} lower bound must be non-negative")
                if log_uniform and low <= 0.0:
                    raise ValueError(f"tracking.{name} log-uniform bounds must be positive")
                setattr(self.tracking, name, [low, high])
            if self.tracking.virtual_contact_randomize_parameters:
                if self.tracking.contact_model != "power_law":
                    raise ValueError(
                        "virtual-contact parameter randomization requires contact_model='power_law'"
                    )
            if not 0.0 <= float(self.tracking.force_sensor_smoothing_factor) <= 1.0:
                raise ValueError("tracking.force_sensor_smoothing_factor must be in [0, 1]")
            if float(self.tracking.torque_sensor_noise_std) < 0.0:
                raise ValueError("tracking.torque_sensor_noise_std must be non-negative")
            if float(self.tracking.torque_sensor_bias_range) < 0.0:
                raise ValueError("tracking.torque_sensor_bias_range must be non-negative")
        if self.tracking.num_future_steps < 1:
            raise ValueError("tracking.num_future_steps must be at least 1")
        self.tool_frame = str(self.tool_frame).strip() or "auto"
        self.robot.spawn.usd_path = f"{ASSET_DIR}/{self.robot_usd_path}"
        self.sim.render_interval = self.decimation

        gain_action_mode = str(self.ctrl.control_gain_action_mode)
        if gain_action_mode not in ("scalar", "per_axis"):
            raise ValueError(
                "ctrl.control_gain_action_mode must be 'scalar' or 'per_axis', "
                f"got {gain_action_mode!r}"
            )
        gain_action_dim = 1 if gain_action_mode == "scalar" else 6
        # Action space: 6 Cartesian delta-pose dims plus optional gain scheduling.
        action_dim = 6 + (gain_action_dim if self.ctrl.control_gains else 0)
        self.action_space = action_dim

        # Proprio obs: ee_pos(3)+ee_quat(4)+optional joint_pos(7)
        # +optional joint_vel(7)+actions(action_dim). Each lookahead pose
        # contributes a (pos_error, axis_angle_error) pair = 6 dims.
        # The six current proportional gains optionally enter the policy. The
        # asymmetric critic can additionally receive the complete sampled
        # controller state (Kp and Kd). The critic also adds 24 privileged dims:
        # payload_mass(1)+payload_com(3)
        # +joint_friction(7)+joint_armature(7)+pos_threshold(3)+rot_threshold(3).
        wrench_dim = 6 if self.tracking.use_full_wrench else 3
        force_feedback_dim = (
            wrench_dim
            if self.tracking.enable_force and self.tracking.force_mode == "virtual_contact"
            else 0
        )
        proprio_dim = (
            7
            + (7 if self.include_joint_angles else 0)
            + (7 if self.include_joint_velocities else 0)
            + action_dim
            + force_feedback_dim
        )
        future_dim = 6 * self.tracking.num_future_steps
        policy_controller_context_dim = 6 if self.controller_context_in_policy else 0
        critic_controller_context_dim = (
            (6 if self.controller_context_in_policy else 12)
            if self.controller_context_in_critic
            else 0
        )
        virtual_contact_privileged_dim = (
            6
            if self.tracking.enable_force and self.tracking.force_mode == "virtual_contact"
            else 0
        )
        privileged_dim = 24 + virtual_contact_privileged_dim
        # Wrench tracking contributes either force3 (legacy) or force3+torque3
        # per future step to both observations.
        force_dim = (
            wrench_dim * self.tracking.num_future_steps
            if self.tracking.enable_force
            else 0
        )
        history = max(1, int(self.obs_history_length))
        # Only proprio is stacked over history; future errors + target wrench +
        # controller gains and privileged dims are appended once from the
        # current frame rather than duplicated through history.
        self.observation_space = (
            proprio_dim * history
            + future_dim
            + force_dim
            + policy_controller_context_dim
        )
        self.state_space = (
            self.observation_space
            + critic_controller_context_dim
            + privileged_dim
        )
