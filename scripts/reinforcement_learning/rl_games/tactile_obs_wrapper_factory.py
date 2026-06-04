from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass(frozen=True)
class TactileObsWrapperSpec:
    name: str  # none|mae|point_mae|ot|double_ot
    tactile_obs_key: str | Sequence[str]
    pc_keys: list[str]
    force_keys: dict[str, str]
    ot_euler_steps: int
    ot_euler_steps_hop2: int | None
    ot_noise_scale: float
    ot_reverse_direction: bool
    use_projection: bool
    projection_source_latent: str | None
    ckpt_point_mae: str | None
    ckpt_mae: str | None
    ckpt_ot: str | None
    ckpt_ot_2: str | None
    ckpt_projection: str | None
    latent_noise_enable: bool
    latent_noise_std: float
    force_input_noise_std: float
    ot_debug_gt_force: bool
    ot_debug_pred_force: bool
    ckpt_force_probe: str | None


def _is_enabled_path(p: Any) -> bool:
    return p is not None and str(p) != "null" and str(p) != ""


def _parse_spec(task_overrides: dict) -> TactileObsWrapperSpec:
    """
    New schema: task_overrides.tactile_obs_wrapper.*
    """
    wrapper_cfg = task_overrides.get("tactile_obs_wrapper")
    if not isinstance(wrapper_cfg, dict):
        raise ValueError(
            "Missing required config: task_overrides.tactile_obs_wrapper. "
            "Add tactile_obs_wrapper to the agent YAML (set name: null to disable)."
        )

    name = str(wrapper_cfg.get("name", "none")).lower()
    from tactile_transfer.utils.policy_keys import normalize_policy_keys  # noqa: WPS433

    tactile_obs_key_cfg = wrapper_cfg.get("tactile_obs_key", "tactile")
    tactile_obs_key_items = normalize_policy_keys(
        tactile_obs_key_cfg,
        arg_name="task_overrides.tactile_obs_wrapper.tactile_obs_key",
    )
    tactile_obs_key: str | Sequence[str]
    if len(tactile_obs_key_items) == 1:
        tactile_obs_key = tactile_obs_key_items[0]
    else:
        tactile_obs_key = tactile_obs_key_items
    pc_keys = list(wrapper_cfg.get("pc_keys") or [])
    force_keys = dict(wrapper_cfg.get("force_keys") or {})
    ot_euler_steps = int(wrapper_cfg.get("ot_euler_steps", 32))
    ot_euler_steps_hop2_raw = wrapper_cfg.get("ot_euler_steps_hop2")
    ot_euler_steps_hop2 = (
        None if ot_euler_steps_hop2_raw is None else int(ot_euler_steps_hop2_raw)
    )
    ot_noise_scale = float(
        wrapper_cfg.get(
            "ot_noise_scale",
            wrapper_cfg.get("rectified_flow_noise_scale", 0.0),
        )
    )
    if ot_noise_scale < 0.0:
        raise ValueError(
            "tactile_obs_wrapper.ot_noise_scale must be non-negative, "
            f"got {ot_noise_scale}."
        )
    ot_reverse_direction = bool(wrapper_cfg.get("ot_reverse_direction", False))
    ot_debug_gt_force = bool(wrapper_cfg.get("debug_gt_force", False))
    ot_debug_pred_force = bool(wrapper_cfg.get("debug_pred_force", False))
    if ot_debug_gt_force and ot_debug_pred_force:
        raise ValueError(
            "tactile_obs_wrapper.debug_gt_force and debug_pred_force cannot both be true."
        )
    use_projection = bool(wrapper_cfg.get("use_projection", False))
    projection_source_latent = wrapper_cfg.get("projection_source_latent")
    if projection_source_latent is not None:
        projection_source_latent = str(projection_source_latent).lower()
    latent_noise_cfg = dict(wrapper_cfg.get("gaussian_dropout") or wrapper_cfg.get("latent_noise") or {})
    latent_noise_enable = bool(latent_noise_cfg.get("enable", False))
    latent_noise_std = float(latent_noise_cfg.get("std", 0.0))
    if latent_noise_std < 0.0:
        raise ValueError(
            "tactile_obs_wrapper.gaussian_dropout.std must be non-negative, "
            f"got {latent_noise_std}."
        )
    force_input_noise_raw = wrapper_cfg.get(
        "force_input_noise_std", wrapper_cfg.get("force_input_noise", 0.0)
    )
    if isinstance(force_input_noise_raw, dict):
        force_input_noise_std = float(force_input_noise_raw.get("std", 0.0))
    else:
        force_input_noise_std = float(force_input_noise_raw)
    if force_input_noise_std < 0.0:
        raise ValueError(
            "tactile_obs_wrapper.force_input_noise_std must be non-negative, "
            f"got {force_input_noise_std}."
        )
    ckpts = dict(wrapper_cfg.get("checkpoints") or {})
    return TactileObsWrapperSpec(
        name=name,
        tactile_obs_key=tactile_obs_key,
        pc_keys=pc_keys,
        force_keys=force_keys,
        ot_euler_steps=ot_euler_steps,
        ot_euler_steps_hop2=ot_euler_steps_hop2,
        ot_noise_scale=ot_noise_scale,
        ot_reverse_direction=ot_reverse_direction,
        use_projection=use_projection,
        projection_source_latent=projection_source_latent,
        ckpt_point_mae=str(ckpts.get("point_mae")) if _is_enabled_path(ckpts.get("point_mae")) else None,
        ckpt_mae=str(ckpts.get("mae")) if _is_enabled_path(ckpts.get("mae")) else None,
        ckpt_ot=str(ckpts.get("ot")) if _is_enabled_path(ckpts.get("ot")) else None,
        ckpt_ot_2=str(ckpts.get("ot_2")) if _is_enabled_path(ckpts.get("ot_2")) else None,
        ckpt_projection=str(ckpts.get("projection")) if _is_enabled_path(ckpts.get("projection")) else None,
        latent_noise_enable=latent_noise_enable,
        latent_noise_std=latent_noise_std,
        force_input_noise_std=force_input_noise_std,
        ot_debug_gt_force=ot_debug_gt_force,
        ot_debug_pred_force=ot_debug_pred_force,
        ckpt_force_probe=str(ckpts.get("force_probe")) if _is_enabled_path(ckpts.get("force_probe")) else None,
    )


def build_tactile_obs_wrapper(
    *,
    env_cfg: Any,
    task_overrides: dict,
    rl_device: str,
) -> Callable[[Any], Any]:
    """
    Returns a function that wraps a created gym env.

    Note: this function mutates env_cfg to enable required observation streams
    (e.g., tactile images or tactile point clouds) BEFORE env creation.
    """
    spec = _parse_spec(task_overrides)
    name = spec.name
    tactile_obs_keys = list(spec.tactile_obs_key) if isinstance(spec.tactile_obs_key, (list, tuple)) else [spec.tactile_obs_key]
    use_right_tactile = any(str(k).strip().lower() == "tactile_right" for k in tactile_obs_keys)
    # Lazy imports to keep the train/play scripts lightweight at import time.
    from tactile_transfer.model import (  # noqa: WPS433
        load_latent_projection_from_checkpoint,
        load_projection_input_norm,
    )
    from tactile_transfer.utils.rl_pointmae_sinkhorn_projection_wrapper import (  # noqa: WPS433
        PointMAEObsWithSinkhornProjectionWrapper,
    )
    from tactile_transfer.utils.rl_pointmae_wrapper import (  # noqa: WPS433
        PointMAEObsWrapper,
        build_pointmae_rl_override_cfg,
        load_pointmae_encoder_from_checkpoint,
    )
    from tactile_transfer.utils.rl_tactile_image_mae_wrapper import (  # noqa: WPS433
        TactileImageMAEObsWrapper,
        load_tactile_image_mae_encoder_from_checkpoint,
    )

    wrap_device = torch.device(rl_device if torch.cuda.is_available() else "cpu")

    if name in ("none", "null", ""):
        return lambda env: env

    if name == "mae":
        if spec.ckpt_mae is None:
            raise ValueError("tactile_obs_wrapper.name='mae' requires checkpoints.mae")
        if hasattr(env_cfg, "enable_tactile_sensor"):
            env_cfg.enable_tactile_sensor = True
        if use_right_tactile and hasattr(env_cfg, "enable_tactile_sensor_right"):
            env_cfg.enable_tactile_sensor_right = True
        image_mae = load_tactile_image_mae_encoder_from_checkpoint(spec.ckpt_mae, wrap_device)
        projection = None
        if spec.use_projection:
            if spec.ckpt_projection is None:
                raise ValueError(
                    "tactile_obs_wrapper.use_projection=true requires checkpoints.projection "
                    "when name='mae'"
                )
            projection = load_latent_projection_from_checkpoint(
                spec.ckpt_projection,
                wrap_device,
                expected_latent_dim=int(image_mae.cfg.encoder_embed_dim)
                * (len(spec.tactile_obs_key) if isinstance(spec.tactile_obs_key, (list, tuple)) else 1),
            )
        return lambda env: TactileImageMAEObsWrapper(
            env,
            image_mae=image_mae,
            device=wrap_device,
            projection=projection,
            tactile_obs_key=spec.tactile_obs_key,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std=spec.latent_noise_std,
        )

    if name == "point_mae":
        if spec.ckpt_point_mae is None:
            raise ValueError("tactile_obs_wrapper.name='point_mae' requires checkpoints.point_mae")

        if hasattr(env_cfg, "include_contact_forces"):
            env_cfg.include_contact_forces = True
        else:
            raise ValueError("include_contact_forces must exist in env_cfg for Point-MAE wrapper")
        if hasattr(env_cfg, "include_tactile_pointclouds"):
            env_cfg.include_tactile_pointclouds = True
        else:
            raise ValueError("include_tactile_pointclouds must exist in env_cfg for Point-MAE wrapper")

        point_cfg = build_pointmae_rl_override_cfg(
            spec.pc_keys, spec.force_keys, force_input_noise_std=spec.force_input_noise_std
        )
        point_mae = load_pointmae_encoder_from_checkpoint(spec.ckpt_point_mae, wrap_device, point_cfg)

        if spec.use_projection:
            if spec.ckpt_projection is None:
                raise ValueError(
                    "tactile_obs_wrapper.use_projection=true requires checkpoints.projection "
                    "(projection is only supported with point_mae)"
                )
            projection = load_latent_projection_from_checkpoint(
                spec.ckpt_projection,
                wrap_device,
                expected_latent_dim=int(point_mae.cfg.embed_dim),
            )
            return lambda env: PointMAEObsWithSinkhornProjectionWrapper(
                env,
                point_mae,
                wrap_device,
                point_cfg,
                projection,
                latent_noise_enable=spec.latent_noise_enable,
                latent_noise_std=spec.latent_noise_std,
            )

        return lambda env: PointMAEObsWrapper(
            env,
            point_mae,
            wrap_device,
            point_cfg,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std=spec.latent_noise_std,
        )

    if name == "ot":
        if spec.ckpt_ot is None:
            raise ValueError("tactile_obs_wrapper.name='ot' requires checkpoints.ot")

        if spec.projection_source_latent is not None and spec.projection_source_latent not in ("point", "image"):
            raise ValueError(
                "tactile_obs_wrapper.projection_source_latent must be 'point' or 'image' "
                f"when tactile_obs_wrapper.name='ot', got {spec.projection_source_latent!r}"
            )

        # OT wrapper lives in tactile_transfer and needs more modules.
        from tactile_transfer.utils.rl_latent_ot_tactile_image_to_point_latent_wrapper import (  # noqa: WPS433
            TactileLatentFlowObsWrapper,
        )
        from tactile_transfer.model import (  # noqa: WPS433
            LatentNormalization,
            RectifiedFlowVelocityConfig,
            VelocityMLP,
        )

        payload = torch.load(spec.ckpt_ot, map_location="cpu")
        if "velocity" not in payload or "latent_normalization" not in payload:
            raise ValueError("OT checkpoint missing required keys: 'velocity' and 'latent_normalization'")

        ckpt_direction = str(payload.get("direction", "image_to_pc"))
        if ckpt_direction not in ("image_to_pc", "pc_to_image"):
            raise ValueError(f"OT checkpoint direction must be 'image_to_pc' or 'pc_to_image', got {ckpt_direction!r}")
        if spec.ot_reverse_direction:
            direction = "pc_to_image" if ckpt_direction == "image_to_pc" else "image_to_pc"
        else:
            direction = ckpt_direction

        latent_dim = int(payload["latent_dim"])
        velocity_cfg_dict = payload.get("velocity_cfg", {}) or {}
        hidden_dims = velocity_cfg_dict.get("hidden_dims", (1024, 1024))
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        velocity_cfg = RectifiedFlowVelocityConfig(
            time_dim=int(velocity_cfg_dict.get("time_dim", 256)),
            proprio_dim=int(velocity_cfg_dict.get("proprio_dim", payload.get("proprio_dim", 0))),
            hidden_dims=tuple(hidden_dims),
        )
        velocity = VelocityMLP(latent_dim=latent_dim, cfg=velocity_cfg).to(wrap_device)
        velocity.load_state_dict(payload["velocity"], strict=True)
        velocity.eval()

        prediction_target = str(payload.get("prediction_target", "velocity"))
        if prediction_target not in ("velocity", "x0"):
            raise ValueError(f"Unsupported OT prediction_target={prediction_target!r} (expected 'velocity' or 'x0').")
        if spec.ot_reverse_direction and prediction_target != "velocity":
            raise ValueError(
                "tactile_obs_wrapper.ot_reverse_direction=true requires an OT checkpoint "
                f"with prediction_target='velocity', got {prediction_target!r}."
            )

        latent_norm = LatentNormalization.from_state_dict(payload["latent_normalization"]).to(wrap_device)

        # Load the source encoder based on direction (checkpoint path comes from shared checkpoints).
        image_mae = None
        point_mae = None
        latent_projection = None
        latent_projection_source = None
        pc_gather_cfg = None

        if spec.use_projection:
            if spec.ckpt_projection is None:
                raise ValueError(
                    "tactile_obs_wrapper.use_projection=true requires checkpoints.projection when name='ot'"
                )
            latent_projection = load_latent_projection_from_checkpoint(
                spec.ckpt_projection,
                wrap_device,
                expected_latent_dim=latent_dim,
            )
            projection_ckpt = torch.load(spec.ckpt_projection, map_location="cpu")
            projection_meta = load_projection_input_norm(projection_ckpt)
            latent_projection_source = str(projection_meta["source_latent"]).lower()
            if spec.projection_source_latent is not None:
                if spec.projection_source_latent != latent_projection_source:
                    raise ValueError(
                        "tactile_obs_wrapper.projection_source_latent does not match projection checkpoint: "
                        f"cfg={spec.projection_source_latent!r} vs ckpt={latent_projection_source!r}"
                    )
                latent_projection_source = spec.projection_source_latent

        if direction == "image_to_pc":
            if spec.ckpt_mae is None:
                raise ValueError("OT direction=image_to_pc requires checkpoints.mae")
            if hasattr(env_cfg, "enable_tactile_sensor"):
                env_cfg.enable_tactile_sensor = True
            if use_right_tactile and hasattr(env_cfg, "enable_tactile_sensor_right"):
                env_cfg.enable_tactile_sensor_right = True
            image_mae = load_tactile_image_mae_encoder_from_checkpoint(spec.ckpt_mae, wrap_device)
            tactile_stream_count = len(spec.tactile_obs_key) if isinstance(spec.tactile_obs_key, (list, tuple)) else 1
            expected_image_latent_dim = int(image_mae.cfg.encoder_embed_dim) * int(tactile_stream_count)
            if expected_image_latent_dim != latent_dim:
                raise ValueError(
                    f"Image MAE latent dim ({int(image_mae.cfg.encoder_embed_dim)} x {tactile_stream_count})="
                    f"{expected_image_latent_dim} != OT latent_dim={latent_dim}."
                )
            if spec.ot_debug_gt_force or spec.ot_debug_pred_force:
                if spec.ckpt_point_mae is None:
                    raise ValueError(
                        "tactile_obs_wrapper.debug_gt_force/debug_pred_force=true requires "
                        "checkpoints.point_mae"
                    )
                if not spec.pc_keys:
                    raise ValueError(
                        "tactile_obs_wrapper.debug_gt_force/debug_pred_force=true requires non-empty pc_keys"
                    )
                if spec.ot_debug_gt_force:
                    if hasattr(env_cfg, "include_contact_forces"):
                        env_cfg.include_contact_forces = True
                    else:
                        raise ValueError(
                            "include_contact_forces must exist in env_cfg for OT debug_gt_force"
                        )
                if hasattr(env_cfg, "include_tactile_pointclouds"):
                    env_cfg.include_tactile_pointclouds = True
                else:
                    raise ValueError(
                        "include_tactile_pointclouds must exist in env_cfg for OT "
                        "debug_gt_force/debug_pred_force"
                    )
                point_cfg = build_pointmae_rl_override_cfg(
                    spec.pc_keys, spec.force_keys, force_input_noise_std=spec.force_input_noise_std
                )
                point_mae = load_pointmae_encoder_from_checkpoint(
                    spec.ckpt_point_mae, wrap_device, point_cfg
                )
                if int(point_mae.cfg.embed_dim) != latent_dim:
                    raise ValueError(
                        f"Point-MAE embed_dim={int(point_mae.cfg.embed_dim)} != OT latent_dim={latent_dim}."
                    )
                pc_gather_cfg = {"pc_keys": list(spec.pc_keys), "force_keys": dict(spec.force_keys)}
        else:
            if spec.ckpt_point_mae is None:
                raise ValueError("OT direction=pc_to_image requires checkpoints.point_mae")
            if not spec.pc_keys:
                raise ValueError("OT direction=pc_to_image requires non-empty pc_keys")
            if hasattr(env_cfg, "include_contact_forces"):
                env_cfg.include_contact_forces = True
            else:
                raise ValueError("include_contact_forces must exist in env_cfg for OT pc_to_image")
            if hasattr(env_cfg, "include_tactile_pointclouds"):
                env_cfg.include_tactile_pointclouds = True
            else:
                raise ValueError("include_tactile_pointclouds must exist in env_cfg for OT pc_to_image")

            point_cfg = build_pointmae_rl_override_cfg(
                spec.pc_keys, spec.force_keys, force_input_noise_std=spec.force_input_noise_std
            )
            point_mae = load_pointmae_encoder_from_checkpoint(spec.ckpt_point_mae, wrap_device, point_cfg)
            if int(point_mae.cfg.embed_dim) != latent_dim:
                raise ValueError(f"Point-MAE embed_dim={int(point_mae.cfg.embed_dim)} != OT latent_dim={latent_dim}.")
            pc_gather_cfg = {"pc_keys": list(spec.pc_keys), "force_keys": dict(spec.force_keys)}

        proprio_mean = None
        proprio_std = None
        if int(velocity_cfg.proprio_dim) > 0:
            expected_proprio_dim = int(velocity_cfg.proprio_dim)
            # Proprio stats (source-side) for normalization.
            # Avoid `a or b` here because these values are tensors and cannot be
            # truth-tested when they contain more than one element.
            if direction == "image_to_pc":
                proprio_mean = payload.get("proprio_img_mean")
                proprio_std = payload.get("proprio_img_std")
            else:
                proprio_mean = payload.get("proprio_pc_mean")
                proprio_std = payload.get("proprio_pc_std")
            if proprio_mean is None:
                proprio_mean = payload.get("proprio_src_mean")
            if proprio_std is None:
                proprio_std = payload.get("proprio_src_std")
            if proprio_mean is None or proprio_std is None:
                raise ValueError(
                    "OT checkpoint missing source proprio mean/std (expected proprio_*_mean/std) "
                    "for proprio-conditioned velocity."
                )

            proprio_mean = proprio_mean.to(wrap_device).float().reshape(-1)
            proprio_std = proprio_std.to(wrap_device).float().reshape(-1)
            if int(proprio_mean.shape[0]) != int(proprio_std.shape[0]):
                raise ValueError(
                    f"OT checkpoint proprio mean/std dim mismatch: mean={proprio_mean.shape[0]} "
                    f"std={proprio_std.shape[0]}."
                )
            if int(proprio_mean.shape[0]) < expected_proprio_dim:
                raise ValueError(
                    "OT checkpoint proprio stats dim is smaller than velocity proprio_dim: "
                    f"stats={proprio_mean.shape[0]} expected={expected_proprio_dim}."
                )
            if int(proprio_mean.shape[0]) > expected_proprio_dim:
                # Backward compatibility: older checkpoints may store full vector_obs stats
                # even when training excluded trailing action dims from conditioning.
                proprio_mean = proprio_mean[:expected_proprio_dim]
                proprio_std = proprio_std[:expected_proprio_dim]

        force_probe = None
        if spec.ot_debug_pred_force:
            if spec.ckpt_force_probe is None:
                raise ValueError(
                    "tactile_obs_wrapper.debug_pred_force=true requires checkpoints.force_probe"
                )
            from tactile_transfer.utils.image_latent_force_probe import (  # noqa: WPS433
                load_image_latent_force_probe_from_checkpoint,
            )

            force_probe = load_image_latent_force_probe_from_checkpoint(
                spec.ckpt_force_probe, wrap_device
            )
            if direction != "image_to_pc":
                raise ValueError("debug_pred_force is only supported for OT direction=image_to_pc")
            if image_mae is None:
                raise ValueError("debug_pred_force requires checkpoints.mae")
            tactile_stream_count = (
                len(spec.tactile_obs_key)
                if isinstance(spec.tactile_obs_key, (list, tuple))
                else 1
            )
            expected_image_latent_dim = int(image_mae.cfg.encoder_embed_dim) * int(tactile_stream_count)
            probe_in_dim = int(force_probe.mu_img.shape[0])
            if probe_in_dim != expected_image_latent_dim:
                raise ValueError(
                    f"Force-probe in_dim={probe_in_dim} does not match image MAE latent dim "
                    f"{expected_image_latent_dim} (embed_dim x tactile streams)."
                )

        wrapper_kwargs = dict(
            image_mae=image_mae,
            velocity=velocity,
            latent_norm=latent_norm,
            device=wrap_device,
            latent_dim=latent_dim,
            tactile_obs_key=spec.tactile_obs_key,
            euler_steps=int(spec.ot_euler_steps),
            noise_scale=float(spec.ot_noise_scale),
            prediction_target=prediction_target,
            direction=direction,
            reverse_training_direction=bool(spec.ot_reverse_direction),
            point_mae=point_mae,
            latent_projection=latent_projection,
            latent_projection_source=latent_projection_source,
            pc_gather_cfg=pc_gather_cfg,
            proprio_src_mean=proprio_mean,
            proprio_src_std=proprio_std,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std=spec.latent_noise_std,
            debug_gt_force=spec.ot_debug_gt_force,
            debug_pred_force=spec.ot_debug_pred_force,
            force_probe=force_probe,
        )

        return lambda env: TactileLatentFlowObsWrapper(env, **wrapper_kwargs)

    if name == "double_ot":
        if spec.ckpt_ot is None:
            raise ValueError("tactile_obs_wrapper.name='double_ot' requires checkpoints.ot")

        from tactile_transfer.utils.rl_latent_ot_tactile_image_to_point_latent_wrapper import (  # noqa: WPS433
            TactileLatentFlowObsWrapper,
        )
        from tactile_transfer.utils.latent_flow_training import (  # noqa: WPS433
            check_rectified_flow_integration_direction,
            hop_spec_for_transition,
            load_rectified_flow_checkpoint,
        )

        hop1 = load_rectified_flow_checkpoint(spec.ckpt_ot, wrap_device)
        hop1_direction = str(hop1["train_direction"])
        hop1_sign = hop_spec_for_transition(
            train_direction=hop1_direction,
            src_modality="point",
            dst_modality="image",
        )[1]
        check_rectified_flow_integration_direction(
            prediction_target=hop1["prediction_target"],
            integration_sign=hop1_sign,
            ckpt_path=hop1["path"],
            hop_label="double_ot hop 1",
        )

        latent_dim = int(hop1["latent_dim"])
        velocity = hop1["velocity"]
        latent_norm = hop1["latent_norm"]
        prediction_target = hop1["prediction_target"]
        payload = hop1["payload"]
        velocity_cfg = hop1["velocity_cfg"]

        velocity_hop2 = None
        latent_norm_hop2 = None
        prediction_target_hop2 = None
        direction_hop2 = None
        ckpt_path_hop2 = None
        if spec.ckpt_ot_2 is not None:
            hop2 = load_rectified_flow_checkpoint(spec.ckpt_ot_2, wrap_device, expected_latent_dim=latent_dim)
            if int(hop2["latent_dim"]) != latent_dim:
                raise ValueError(
                    f"checkpoints.ot_2 latent_dim={hop2['latent_dim']} != hop-1 latent_dim={latent_dim}."
                )
            hop2_sign = hop_spec_for_transition(
                train_direction=hop2["train_direction"],
                src_modality="image",
                dst_modality="point",
            )[1]
            check_rectified_flow_integration_direction(
                prediction_target=hop2["prediction_target"],
                integration_sign=hop2_sign,
                ckpt_path=hop2["path"],
                hop_label="double_ot hop 2",
            )
            velocity_hop2 = hop2["velocity"]
            latent_norm_hop2 = hop2["latent_norm"]
            prediction_target_hop2 = hop2["prediction_target"]
            direction_hop2 = hop2["train_direction"]
            ckpt_path_hop2 = str(hop2["path"])
        else:
            hop2_sign = hop_spec_for_transition(
                train_direction=hop1_direction,
                src_modality="image",
                dst_modality="point",
            )[1]
            check_rectified_flow_integration_direction(
                prediction_target=prediction_target,
                integration_sign=hop2_sign,
                ckpt_path=hop1["path"],
                hop_label="double_ot hop 2",
            )

        if spec.ckpt_point_mae is None:
            raise ValueError("tactile_obs_wrapper.name='double_ot' requires checkpoints.point_mae")
        if not spec.pc_keys:
            raise ValueError("tactile_obs_wrapper.name='double_ot' requires non-empty pc_keys")
        if hasattr(env_cfg, "include_contact_forces"):
            env_cfg.include_contact_forces = True
        else:
            raise ValueError("include_contact_forces must exist in env_cfg for double_ot")
        if hasattr(env_cfg, "include_tactile_pointclouds"):
            env_cfg.include_tactile_pointclouds = True
        else:
            raise ValueError("include_tactile_pointclouds must exist in env_cfg for double_ot")

        point_cfg = build_pointmae_rl_override_cfg(
            spec.pc_keys, spec.force_keys, force_input_noise_std=spec.force_input_noise_std
        )
        point_mae = load_pointmae_encoder_from_checkpoint(spec.ckpt_point_mae, wrap_device, point_cfg)
        if int(point_mae.cfg.embed_dim) != latent_dim:
            raise ValueError(
                f"Point-MAE embed_dim={int(point_mae.cfg.embed_dim)} != OT latent_dim={latent_dim}."
            )
        pc_gather_cfg = {"pc_keys": list(spec.pc_keys), "force_keys": dict(spec.force_keys)}

        latent_projection = None
        latent_projection_source = None
        if spec.use_projection:
            if spec.ckpt_projection is None:
                raise ValueError(
                    "tactile_obs_wrapper.use_projection=true requires checkpoints.projection when name='double_ot'"
                )
            latent_projection = load_latent_projection_from_checkpoint(
                spec.ckpt_projection,
                wrap_device,
                expected_latent_dim=latent_dim,
            )
            projection_ckpt = torch.load(spec.ckpt_projection, map_location="cpu")
            projection_meta = load_projection_input_norm(projection_ckpt)
            latent_projection_source = str(projection_meta["source_latent"]).lower()
            if spec.projection_source_latent is not None:
                if spec.projection_source_latent != latent_projection_source:
                    raise ValueError(
                        "tactile_obs_wrapper.projection_source_latent does not match projection checkpoint: "
                        f"cfg={spec.projection_source_latent!r} vs ckpt={latent_projection_source!r}"
                    )
                latent_projection_source = spec.projection_source_latent
            if latent_projection_source != "point":
                raise ValueError(
                    "double_ot only supports projection on the point branch; "
                    f"projection checkpoint source_latent={latent_projection_source!r}."
                )

        proprio_img_mean = None
        proprio_img_std = None
        proprio_pc_mean = None
        proprio_pc_std = None
        if int(velocity_cfg.proprio_dim) > 0:
            expected_proprio_dim = int(velocity_cfg.proprio_dim)
            proprio_img_mean = payload.get("proprio_img_mean")
            proprio_img_std = payload.get("proprio_img_std")
            proprio_pc_mean = payload.get("proprio_pc_mean")
            proprio_pc_std = payload.get("proprio_pc_std")
            if proprio_img_mean is None or proprio_img_std is None:
                raise ValueError(
                    "double_ot hop-1 checkpoint missing proprio_img_mean/proprio_img_std."
                )
            if proprio_pc_mean is None or proprio_pc_std is None:
                raise ValueError(
                    "double_ot hop-1 checkpoint missing proprio_pc_mean/proprio_pc_std."
                )
            for label, mean_t, std_t in (
                ("proprio_img", proprio_img_mean, proprio_img_std),
                ("proprio_pc", proprio_pc_mean, proprio_pc_std),
            ):
                mean_t = mean_t.to(wrap_device).float().reshape(-1)
                std_t = std_t.to(wrap_device).float().reshape(-1)
                if int(mean_t.shape[0]) != int(std_t.shape[0]):
                    raise ValueError(
                        f"OT checkpoint {label} mean/std dim mismatch: "
                        f"mean={mean_t.shape[0]} std={std_t.shape[0]}."
                    )
                if int(mean_t.shape[0]) < expected_proprio_dim:
                    raise ValueError(
                        f"OT checkpoint {label} stats dim is smaller than velocity proprio_dim: "
                        f"stats={mean_t.shape[0]} expected={expected_proprio_dim}."
                    )
                if int(mean_t.shape[0]) > expected_proprio_dim:
                    mean_t = mean_t[:expected_proprio_dim]
                    std_t = std_t[:expected_proprio_dim]
                if label == "proprio_img":
                    proprio_img_mean, proprio_img_std = mean_t, std_t
                else:
                    proprio_pc_mean, proprio_pc_std = mean_t, std_t

        return lambda env: TactileLatentFlowObsWrapper(
            env,
            image_mae=None,
            velocity=velocity,
            latent_norm=latent_norm,
            device=wrap_device,
            latent_dim=latent_dim,
            euler_steps=int(spec.ot_euler_steps),
            euler_steps_hop2=spec.ot_euler_steps_hop2,
            noise_scale=float(spec.ot_noise_scale),
            prediction_target=prediction_target,
            direction=hop1_direction,
            reverse_training_direction=False,
            point_mae=point_mae,
            latent_projection=latent_projection,
            latent_projection_source=latent_projection_source,
            pc_gather_cfg=pc_gather_cfg,
            proprio_src_mean=proprio_img_mean,
            proprio_src_std=proprio_img_std,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std=spec.latent_noise_std,
            double_ot=True,
            proprio_pc_mean=proprio_pc_mean,
            proprio_pc_std=proprio_pc_std,
            velocity_hop2=velocity_hop2,
            latent_norm_hop2=latent_norm_hop2,
            prediction_target_hop2=prediction_target_hop2,
            direction_hop2=direction_hop2,
            ckpt_path_hop2=ckpt_path_hop2,
        )

    raise ValueError(
        f"Unsupported tactile_obs_wrapper.name={name!r}. Expected null|mae|point_mae|ot|double_ot."
    )

