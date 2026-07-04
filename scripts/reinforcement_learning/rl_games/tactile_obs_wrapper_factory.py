from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass(frozen=True)
class TactileObsWrapperSpec:
    name: str  # none|mae|point_mae|ot|double_ot|contact_point
    tactile_obs_key: str | Sequence[str]
    pc_keys: list[str]
    force_keys: dict[str, str]
    ot_euler_steps: int
    ot_euler_steps_hop2: int | None
    ot_reverse_direction: bool
    use_projection: bool
    projection_source_latent: str | None
    ckpt_point_mae: str | None
    ckpt_mae: str | None
    ckpt_ot: str | None
    ckpt_ot_2: str | None
    ckpt_projection: str | None
    force_input_noise_std: float


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
    ot_reverse_direction = bool(wrapper_cfg.get("ot_reverse_direction", False))
    use_projection = bool(wrapper_cfg.get("use_projection", False))
    projection_source_latent = wrapper_cfg.get("projection_source_latent")
    if projection_source_latent is not None:
        projection_source_latent = str(projection_source_latent).lower()
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
        ot_reverse_direction=ot_reverse_direction,
        use_projection=use_projection,
        projection_source_latent=projection_source_latent,
        ckpt_point_mae=str(ckpts.get("point_mae")) if _is_enabled_path(ckpts.get("point_mae")) else None,
        ckpt_mae=str(ckpts.get("mae")) if _is_enabled_path(ckpts.get("mae")) else None,
        ckpt_ot=str(ckpts.get("ot")) if _is_enabled_path(ckpts.get("ot")) else None,
        ckpt_ot_2=str(ckpts.get("ot_2")) if _is_enabled_path(ckpts.get("ot_2")) else None,
        ckpt_projection=str(ckpts.get("projection")) if _is_enabled_path(ckpts.get("projection")) else None,
        force_input_noise_std=force_input_noise_std,
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

    if name == "contact_point":
        if hasattr(env_cfg, "include_contact_points"):
            env_cfg.include_contact_points = True
        else:
            raise ValueError("include_contact_points must exist in env_cfg for contact_point wrapper")
        from tactile_transfer.utils.rl_contact_point_wrapper import ContactPointObsWrapper  # noqa: WPS433

        return lambda env: ContactPointObsWrapper(env)

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
            )

        return lambda env: PointMAEObsWrapper(
            env,
            point_mae,
            wrap_device,
            point_cfg,
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
        from tactile_transfer.utils.latent_flow_training import (  # noqa: WPS433
            cond_dim_from_checkpoint,
            flow_mode_from_checkpoint,
        )

        flow_mode = flow_mode_from_checkpoint(payload)
        model_cond_dim = cond_dim_from_checkpoint(velocity_cfg_dict, payload)
        if flow_mode == "noise_to_data_latent_cond":
            if model_cond_dim != latent_dim:
                raise ValueError(
                    "noise_to_data_latent_cond OT checkpoint requires cond_dim == latent_dim, "
                    f"got cond_dim={model_cond_dim}, latent_dim={latent_dim}."
                )
            if spec.ot_reverse_direction:
                raise ValueError(
                    "tactile_obs_wrapper.ot_reverse_direction=true is incompatible with "
                    "flow_mode='noise_to_data_latent_cond' (source latent must match train direction)."
                )

        velocity_cfg = RectifiedFlowVelocityConfig(
            time_dim=int(velocity_cfg_dict.get("time_dim", 256)),
            cond_dim=model_cond_dim,
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
        if int(velocity_cfg.cond_dim) > 0 and flow_mode != "noise_to_data_latent_cond":
            expected_proprio_dim = int(velocity_cfg.cond_dim)
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
                    "OT checkpoint proprio stats dim is smaller than velocity cond_dim: "
                    f"stats={proprio_mean.shape[0]} expected={expected_proprio_dim}."
                )
            if int(proprio_mean.shape[0]) > expected_proprio_dim:
                # Backward compatibility: older checkpoints may store full vector_obs stats
                # even when training excluded trailing action dims from conditioning.
                proprio_mean = proprio_mean[:expected_proprio_dim]
                proprio_std = proprio_std[:expected_proprio_dim]

        wrapper_kwargs = dict(
            image_mae=image_mae,
            velocity=velocity,
            latent_norm=latent_norm,
            device=wrap_device,
            latent_dim=latent_dim,
            tactile_obs_key=spec.tactile_obs_key,
            euler_steps=int(spec.ot_euler_steps),
            prediction_target=prediction_target,
            direction=direction,
            reverse_training_direction=bool(spec.ot_reverse_direction),
            point_mae=point_mae,
            latent_projection=latent_projection,
            latent_projection_source=latent_projection_source,
            pc_gather_cfg=pc_gather_cfg,
            proprio_src_mean=proprio_mean,
            proprio_src_std=proprio_std,
            flow_mode=flow_mode,
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
        flow_mode_hop1 = str(hop1["flow_mode"])
        hop1_direction = str(hop1["train_direction"])
        if flow_mode_hop1 == "noise_to_data_latent_cond":
            if hop1_direction != "pc_to_image":
                raise ValueError(
                    "double_ot hop-1 with flow_mode='noise_to_data_latent_cond' requires "
                    f"train_direction='pc_to_image', got {hop1_direction!r}."
                )
            hop1_sign = 1.0
        else:
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
        hop2_train_direction = None
        flow_mode_hop2 = flow_mode_hop1
        if spec.ckpt_ot_2 is not None:
            hop2 = load_rectified_flow_checkpoint(spec.ckpt_ot_2, wrap_device)
            flow_mode_hop2 = str(hop2["flow_mode"])
            if int(hop2["latent_dim"]) != latent_dim:
                raise ValueError(
                    f"double_ot hop-2 latent_dim={int(hop2['latent_dim'])} != hop-1 latent_dim={latent_dim}."
                )
            hop2_train_direction = str(hop2["train_direction"])
            if flow_mode_hop2 == "noise_to_data_latent_cond":
                if hop2_train_direction != "image_to_pc":
                    raise ValueError(
                        "double_ot hop-2 with flow_mode='noise_to_data_latent_cond' requires "
                        f"train_direction='image_to_pc', got {hop2_train_direction!r}."
                    )
                hop2_sign = 1.0
            else:
                hop2_sign = hop_spec_for_transition(
                    train_direction=hop2_train_direction,
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
            hop2_payload = hop2["payload"]
            hop2_velocity_cfg = hop2["velocity_cfg"]
        else:
            if flow_mode_hop1 == "noise_to_data_latent_cond":
                raise ValueError(
                    "tactile_obs_wrapper.name='double_ot' with flow_mode='noise_to_data_latent_cond' on hop 1 "
                    "requires checkpoints.ot_2 (reverse integration is not supported)."
                )
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
            hop2_payload = payload
            hop2_velocity_cfg = velocity_cfg

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

        hop1_proprio_cond_dim = (
            0 if flow_mode_hop1 == "noise_to_data_latent_cond" else int(velocity_cfg.cond_dim)
        )
        hop2_proprio_cond_dim = (
            0 if flow_mode_hop2 == "noise_to_data_latent_cond" else int(hop2_velocity_cfg.cond_dim)
        )
        if flow_mode_hop2 == "noise_to_data_latent_cond" and spec.ckpt_ot_2 is None:
            raise ValueError(
                "double_ot hop 2 with flow_mode='noise_to_data_latent_cond' requires checkpoints.ot_2."
            )
        if (
            hop1_proprio_cond_dim > 0
            and hop2_proprio_cond_dim > 0
            and hop1_proprio_cond_dim != hop2_proprio_cond_dim
        ):
            raise ValueError(
                "double_ot hop-1 and hop-2 proprio cond_dim must match when both use data_to_data proprio "
                f"conditioning: hop1={hop1_proprio_cond_dim}, hop2={hop2_proprio_cond_dim}."
            )

        proprio_img_mean = None
        proprio_img_std = None
        proprio_pc_mean = None
        proprio_pc_std = None
        if hop1_proprio_cond_dim > 0:
            proprio_pc_mean = payload.get("proprio_pc_mean")
            proprio_pc_std = payload.get("proprio_pc_std")
            if proprio_pc_mean is None or proprio_pc_std is None:
                raise ValueError(
                    "double_ot hop-1 checkpoint missing proprio_pc_mean/proprio_pc_std."
                )
            proprio_pc_mean = proprio_pc_mean.to(wrap_device).float().reshape(-1)
            proprio_pc_std = proprio_pc_std.to(wrap_device).float().reshape(-1)
            if int(proprio_pc_mean.shape[0]) != int(proprio_pc_std.shape[0]):
                raise ValueError(
                    "OT checkpoint proprio_pc mean/std dim mismatch: "
                    f"mean={proprio_pc_mean.shape[0]} std={proprio_pc_std.shape[0]}."
                )
            if int(proprio_pc_mean.shape[0]) < hop1_proprio_cond_dim:
                raise ValueError(
                    "OT checkpoint proprio_pc stats dim is smaller than hop-1 velocity cond_dim: "
                    f"stats={proprio_pc_mean.shape[0]} expected={hop1_proprio_cond_dim}."
                )
            if int(proprio_pc_mean.shape[0]) > hop1_proprio_cond_dim:
                proprio_pc_mean = proprio_pc_mean[:hop1_proprio_cond_dim]
                proprio_pc_std = proprio_pc_std[:hop1_proprio_cond_dim]

        if hop2_proprio_cond_dim > 0:
            proprio_img_mean = hop2_payload.get("proprio_img_mean")
            proprio_img_std = hop2_payload.get("proprio_img_std")
            if proprio_img_mean is None or proprio_img_std is None:
                raise ValueError(
                    "double_ot hop-2 checkpoint missing proprio_img_mean/proprio_img_std "
                    f"({'checkpoints.ot_2' if spec.ckpt_ot_2 is not None else 'checkpoints.ot'})."
                )
            proprio_img_mean = proprio_img_mean.to(wrap_device).float().reshape(-1)
            proprio_img_std = proprio_img_std.to(wrap_device).float().reshape(-1)
            if int(proprio_img_mean.shape[0]) != int(proprio_img_std.shape[0]):
                raise ValueError(
                    "OT checkpoint proprio_img mean/std dim mismatch: "
                    f"mean={proprio_img_mean.shape[0]} std={proprio_img_std.shape[0]}."
                )
            if int(proprio_img_mean.shape[0]) < hop2_proprio_cond_dim:
                raise ValueError(
                    "OT checkpoint proprio_img stats dim is smaller than hop-2 velocity cond_dim: "
                    f"stats={proprio_img_mean.shape[0]} expected={hop2_proprio_cond_dim}."
                )
            if int(proprio_img_mean.shape[0]) > hop2_proprio_cond_dim:
                proprio_img_mean = proprio_img_mean[:hop2_proprio_cond_dim]
                proprio_img_std = proprio_img_std[:hop2_proprio_cond_dim]

        return lambda env: TactileLatentFlowObsWrapper(
            env,
            image_mae=None,
            velocity=velocity,
            latent_norm=latent_norm,
            device=wrap_device,
            latent_dim=latent_dim,
            euler_steps=int(spec.ot_euler_steps),
            euler_steps_hop2=spec.ot_euler_steps_hop2,
            prediction_target=prediction_target,
            direction=hop1_direction,
            reverse_training_direction=False,
            point_mae=point_mae,
            latent_projection=latent_projection,
            latent_projection_source=latent_projection_source,
            pc_gather_cfg=pc_gather_cfg,
            proprio_src_mean=proprio_img_mean,
            proprio_src_std=proprio_img_std,
            double_ot=True,
            proprio_pc_mean=proprio_pc_mean,
            proprio_pc_std=proprio_pc_std,
            flow_mode=flow_mode_hop1,
            flow_mode_hop2=flow_mode_hop2,
            velocity_hop2=velocity_hop2,
            latent_norm_hop2=latent_norm_hop2,
            prediction_target_hop2=prediction_target_hop2,
            hop2_train_direction=hop2_train_direction,
        )

    raise ValueError(
        f"Unsupported tactile_obs_wrapper.name={name!r}. Expected null|mae|point_mae|ot|double_ot|contact_point."
    )


def resolve_policy_latent_dim(task_overrides: dict) -> tuple[int, str]:
    """Return ``(appended_latent_dim, wrapper_name)`` for the active tactile obs wrapper.

    When the wrapper is disabled (``name: null``), returns ``(0, name)``.
    """
    from tactile_transfer.model.point_mae import point_mae_config_from_checkpoint_dict  # noqa: WPS433
    from tactile_transfer.utils.policy_keys import normalize_policy_keys  # noqa: WPS433

    spec = _parse_spec(task_overrides)
    name = spec.name
    if name in ("none", "null", ""):
        return 0, name

    if name == "mae":
        if spec.ckpt_mae is None:
            raise ValueError("tactile_obs_wrapper.name='mae' requires checkpoints.mae")
        ckpt = torch.load(spec.ckpt_mae, map_location="cpu", weights_only=False)
        embed_dim = int(ckpt["config"]["encoder_embed_dim"])
        stream_count = len(
            normalize_policy_keys(spec.tactile_obs_key, arg_name="tactile_obs_wrapper.tactile_obs_key")
        )
        return embed_dim * stream_count, name

    if name == "point_mae":
        if spec.ckpt_point_mae is None:
            raise ValueError("tactile_obs_wrapper.name='point_mae' requires checkpoints.point_mae")
        ckpt = torch.load(spec.ckpt_point_mae, map_location="cpu", weights_only=False)
        embed_dim = int(point_mae_config_from_checkpoint_dict(ckpt["config"]).embed_dim)
        return embed_dim, name

    if name == "contact_point":
        from tactile_transfer.utils.rl_contact_point_wrapper import CONTACT_POINT_OBS_DIM  # noqa: WPS433

        return CONTACT_POINT_OBS_DIM, name

    if name in ("ot", "double_ot"):
        if spec.ckpt_ot is None:
            raise ValueError(f"tactile_obs_wrapper.name={name!r} requires checkpoints.ot")
        payload = torch.load(spec.ckpt_ot, map_location="cpu", weights_only=False)
        if "latent_dim" not in payload:
            raise ValueError(f"OT checkpoint missing 'latent_dim': {spec.ckpt_ot}")
        return int(payload["latent_dim"]), name

    raise ValueError(
        f"Unsupported tactile_obs_wrapper.name={name!r}. Expected null|mae|point_mae|ot|double_ot|contact_point."
    )

