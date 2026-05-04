from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class TactileObsWrapperSpec:
    name: str  # none|mae|point_mae|ot
    tactile_obs_key: str
    pc_keys: list[str]
    force_keys: dict[str, str]
    ot_euler_steps: int
    use_projection: bool
    projection_source_latent: str | None
    ckpt_point_mae: str | None
    ckpt_mae: str | None
    ckpt_ot: str | None
    ckpt_projection: str | None
    latent_noise_enable: bool
    latent_noise_std_min: float
    latent_noise_std_max: float


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
    tactile_obs_key = str(wrapper_cfg.get("tactile_obs_key", "tactile"))
    pc_keys = list(wrapper_cfg.get("pc_keys") or [])
    force_keys = dict(wrapper_cfg.get("force_keys") or {})
    ot_euler_steps = int(wrapper_cfg.get("ot_euler_steps", 32))
    use_projection = bool(wrapper_cfg.get("use_projection", False))
    projection_source_latent = wrapper_cfg.get("projection_source_latent")
    if projection_source_latent is not None:
        projection_source_latent = str(projection_source_latent).lower()
    latent_noise_cfg = dict(wrapper_cfg.get("latent_noise") or {})
    latent_noise_enable = bool(latent_noise_cfg.get("enable", False))
    latent_noise_std_range = latent_noise_cfg.get("std_range")
    if not isinstance(latent_noise_std_range, (list, tuple)) or len(latent_noise_std_range) != 2:
        raise ValueError("tactile_obs_wrapper.latent_noise.std_range is required and must be [min, max].")
    latent_noise_std_min = float(latent_noise_std_range[0])
    latent_noise_std_max = float(latent_noise_std_range[1])
    if latent_noise_std_min < 0.0 or latent_noise_std_max < 0.0:
        raise ValueError("tactile_obs_wrapper.latent_noise std bounds must be non-negative.")
    if latent_noise_std_min > latent_noise_std_max:
        raise ValueError("tactile_obs_wrapper.latent_noise.std_range must satisfy min <= max.")
    ckpts = dict(wrapper_cfg.get("checkpoints") or {})
    return TactileObsWrapperSpec(
        name=name,
        tactile_obs_key=tactile_obs_key,
        pc_keys=pc_keys,
        force_keys=force_keys,
        ot_euler_steps=ot_euler_steps,
        use_projection=use_projection,
        projection_source_latent=projection_source_latent,
        ckpt_point_mae=str(ckpts.get("point_mae")) if _is_enabled_path(ckpts.get("point_mae")) else None,
        ckpt_mae=str(ckpts.get("mae")) if _is_enabled_path(ckpts.get("mae")) else None,
        ckpt_ot=str(ckpts.get("ot")) if _is_enabled_path(ckpts.get("ot")) else None,
        ckpt_projection=str(ckpts.get("projection")) if _is_enabled_path(ckpts.get("projection")) else None,
        latent_noise_enable=latent_noise_enable,
        latent_noise_std_min=latent_noise_std_min,
        latent_noise_std_max=latent_noise_std_max,
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
                expected_latent_dim=int(image_mae.cfg.encoder_embed_dim),
            )
        return lambda env: TactileImageMAEObsWrapper(
            env,
            image_mae=image_mae,
            device=wrap_device,
            projection=projection,
            tactile_obs_key=spec.tactile_obs_key,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std_min=spec.latent_noise_std_min,
            latent_noise_std_max=spec.latent_noise_std_max,
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

        point_cfg = {
            "pc_keys": spec.pc_keys,
            "force_keys": spec.force_keys,
        }
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
                latent_noise_std_min=spec.latent_noise_std_min,
                latent_noise_std_max=spec.latent_noise_std_max,
            )

        return lambda env: PointMAEObsWrapper(
            env,
            point_mae,
            wrap_device,
            point_cfg,
            latent_noise_enable=spec.latent_noise_enable,
            latent_noise_std_min=spec.latent_noise_std_min,
            latent_noise_std_max=spec.latent_noise_std_max,
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

        direction = str(payload.get("direction", "image_to_pc"))
        if direction not in ("image_to_pc", "pc_to_image"):
            raise ValueError(f"OT checkpoint direction must be 'image_to_pc' or 'pc_to_image', got {direction!r}")

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
            image_mae = load_tactile_image_mae_encoder_from_checkpoint(spec.ckpt_mae, wrap_device)
            if int(image_mae.cfg.encoder_embed_dim) != latent_dim:
                raise ValueError(
                    f"Image MAE encoder_embed_dim={int(image_mae.cfg.encoder_embed_dim)} != OT latent_dim={latent_dim}."
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

            point_cfg = {"pc_keys": spec.pc_keys, "force_keys": spec.force_keys}
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
            point_mae=point_mae,
            latent_projection=latent_projection,
            latent_projection_source=latent_projection_source,
            pc_gather_cfg=pc_gather_cfg,
            proprio_src_mean=proprio_mean,
            proprio_src_std=proprio_std,
        )

        return lambda env: TactileLatentFlowObsWrapper(env, **wrapper_kwargs)

    raise ValueError(f"Unsupported tactile_obs_wrapper.name={name!r}. Expected null|mae|point_mae|ot.")

