# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacSL-style tactile image augmentation.

Batched, pure-torch image augmentation for visuotactile sensor observations.
Two augmentation granularities are supported, matching TacSL:

* **Episode-level** (stateful): color jitter (brightness/contrast/saturation/hue),
  random-resized-crop, and optional RGB channel permutation are sampled per env
  (and per sensor) at episode reset via :meth:`TactileImageAugmentor.reset`, and
  held fixed for every frame of that episode. This models stable sensor-specific
  appearance: LED color/intensity, elastomer tint, camera mounting shift/zoom.
* **Timestep-level** (stateless): a smaller color jitter resampled independently
  on every :meth:`TactileImageAugmentor.apply` call, modeling frame-to-frame
  exposure/lighting fluctuation.

Inputs to :meth:`apply` are float tensors in ``[0, 1]`` shaped ``[B, 3, H, W]``
(or ``[B, T, 3, H, W]``, folded internally); outputs preserve shape, dtype,
device, and the ``[0, 1]`` range. All sampling and transforms are vectorized on
the input tensor's device.

This module intentionally has no Isaac Sim / Isaac Lab imports so it can be
unit-tested without launching the simulator.
"""

from __future__ import annotations

import torch
from torchvision.ops import roi_align


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 3, H, W]`` RGB in [0, 1] to HSV with h in [0, 1)."""
    r, g, b = img.unbind(dim=1)
    maxc = img.amax(dim=1)
    minc = img.amin(dim=1)
    v = maxc
    deltac = maxc - minc
    s = torch.where(maxc > 0, deltac / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    deltac_safe = deltac.clamp(min=1e-8)
    rc = (maxc - r) / deltac_safe
    gc = (maxc - g) / deltac_safe
    bc = (maxc - b) / deltac_safe
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(deltac > 0, h, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 3, H, W]`` HSV (h in [0, 1)) to RGB in [0, 1]."""
    h, s, v = img.unbind(dim=1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.to(torch.long) % 6
    mask = i.unsqueeze(1) == torch.arange(6, device=img.device).view(1, 6, 1, 1)
    r = torch.stack([v, q, p, p, t, v], dim=1)
    g = torch.stack([t, v, v, q, p, p], dim=1)
    b = torch.stack([p, p, t, v, v, q], dim=1)
    return torch.stack([(r * mask).sum(1), (g * mask).sum(1), (b * mask).sum(1)], dim=1)


def _luminance(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel grayscale luminance, ``[B, 1, H, W]``."""
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)


def _apply_color_jitter(
    img: torch.Tensor,
    b_factor: torch.Tensor | None,
    c_factor: torch.Tensor | None,
    s_factor: torch.Tensor | None,
    h_delta: torch.Tensor | None,
) -> torch.Tensor:
    """Apply per-env B/C/S/H jitter. Factors are ``[B]`` tensors (1.0 = identity),
    ``h_delta`` is an additive hue shift in fractions of the hue circle."""
    if b_factor is not None:
        img = img * b_factor.view(-1, 1, 1, 1)
    if c_factor is not None:
        mean = _luminance(img).mean(dim=(2, 3), keepdim=True)
        img = (img - mean) * c_factor.view(-1, 1, 1, 1) + mean
    if s_factor is not None:
        gray = _luminance(img)
        img = (img - gray) * s_factor.view(-1, 1, 1, 1) + gray
    img = img.clamp(0.0, 1.0)
    if h_delta is not None:
        hsv = _rgb_to_hsv(img)
        hsv[:, 0] = (hsv[:, 0] + h_delta.view(-1, 1, 1)) % 1.0
        img = _hsv_to_rgb(hsv)
    return img.clamp(0.0, 1.0)


class TactileImageAugmentor:
    """Stateful TacSL-style augmentor for tactile images.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device all state lives on (must match the image tensors).
        img_hw: (height, width) of the tactile images.
        use_ep_aug: Enable episode-level color jitter + random resized crop.
        use_t_aug: Enable timestep-level color jitter.
        randomize_color_channel: Enable per-episode RGB channel permutation.
        ep_bcsh: Episode-level [brightness, contrast, saturation, hue] levels.
        t_bcsh: Timestep-level [brightness, contrast, saturation, hue] levels.
        crop_scale: [min, max] area fraction for the episode-level crop.
        aspect_ratio: [min, max] aspect-ratio multiplier for the crop.
        sensors: Identifiers of independently-augmented sensors.
        share_lr: If True, all sensors share the first sensor's episode transform.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        img_hw: tuple[int, int],
        use_ep_aug: bool = True,
        use_t_aug: bool = True,
        randomize_color_channel: bool = False,
        ep_bcsh: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.2),
        t_bcsh: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02),
        crop_scale: tuple[float, float] = (0.85, 1.0),
        aspect_ratio: tuple[float, float] = (0.9, 1.1),
        sensors: tuple[str, ...] = ("left", "right"),
        share_lr: bool = False,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.img_hw = tuple(img_hw)
        self.use_ep_aug = use_ep_aug
        self.use_t_aug = use_t_aug
        self.randomize_color_channel = randomize_color_channel
        self.ep_bcsh = tuple(float(x) for x in ep_bcsh)
        self.t_bcsh = tuple(float(x) for x in t_bcsh)
        self.crop_scale = tuple(float(x) for x in crop_scale)
        self.aspect_ratio = tuple(float(x) for x in aspect_ratio)
        self.sensors = tuple(sensors)
        self.share_lr = share_lr

        # Per-sensor episode-level state buffers.
        n = num_envs
        self._state: dict[str, dict[str, torch.Tensor]] = {}
        for s in self.sensors:
            self._state[s] = {
                "b": torch.ones(n, device=self.device),
                "c": torch.ones(n, device=self.device),
                "s": torch.ones(n, device=self.device),
                "h": torch.zeros(n, device=self.device),
                # crop boxes as (x1, y1, x2, y2) in pixels; default = full image
                "crop": torch.tensor(
                    [0.0, 0.0, self.img_hw[1], self.img_hw[0]], device=self.device
                ).repeat(n, 1),
                "perm": torch.arange(3, device=self.device).repeat(n, 1),
            }
        self.reset(torch.arange(num_envs, device=self.device))

    @property
    def enabled(self) -> bool:
        return self.use_ep_aug or self.use_t_aug or self.randomize_color_channel

    def _sample_factor(self, level: float, n: int) -> torch.Tensor | None:
        """Multiplicative jitter factor U[max(0, 1-level), 1+level]; None if level==0."""
        if level <= 0.0:
            return None
        low, high = max(0.0, 1.0 - level), 1.0 + level
        return torch.rand(n, device=self.device) * (high - low) + low

    def _sample_hue(self, level: float, n: int) -> torch.Tensor | None:
        """Additive hue delta U[-level, level]; None if level==0."""
        if level <= 0.0:
            return None
        return (torch.rand(n, device=self.device) * 2.0 - 1.0) * level

    def reset(self, env_ids: torch.Tensor) -> None:
        """Resample episode-level transforms for the given env ids."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        n = env_ids.numel()
        if n == 0:
            return
        H, W = self.img_hw
        for i, sensor in enumerate(self.sensors):
            st = self._state[sensor]
            if self.share_lr and i > 0:
                src = self._state[self.sensors[0]]
                for key in st:
                    st[key][env_ids] = src[key][env_ids]
                continue
            if self.use_ep_aug:
                b, c, s, h = self.ep_bcsh
                for key, level in (("b", b), ("c", c), ("s", s)):
                    fac = self._sample_factor(level, n)
                    st[key][env_ids] = fac if fac is not None else 1.0
                hue = self._sample_hue(h, n)
                st["h"][env_ids] = hue if hue is not None else 0.0
                # Random resized crop: sample area scale and aspect-ratio multiplier,
                # derive crop height/width, then a random top-left position.
                scale = (
                    torch.rand(n, device=self.device) * (self.crop_scale[1] - self.crop_scale[0])
                    + self.crop_scale[0]
                )
                ar = (
                    torch.rand(n, device=self.device)
                    * (self.aspect_ratio[1] - self.aspect_ratio[0])
                    + self.aspect_ratio[0]
                )
                crop_h = (H * torch.sqrt(scale / ar)).clamp(max=H)
                crop_w = (W * torch.sqrt(scale * ar)).clamp(max=W)
                y1 = torch.rand(n, device=self.device) * (H - crop_h)
                x1 = torch.rand(n, device=self.device) * (W - crop_w)
                st["crop"][env_ids] = torch.stack([x1, y1, x1 + crop_w, y1 + crop_h], dim=1)
            if self.randomize_color_channel:
                st["perm"][env_ids] = torch.argsort(torch.rand(n, 3, device=self.device), dim=1)

    def apply(self, img: torch.Tensor, sensor: str = "left") -> torch.Tensor:
        """Augment a batch of tactile images.

        Args:
            img: ``[B, 3, H, W]`` or ``[B, T, 3, H, W]`` float tensor in [0, 1],
                where B == num_envs (row i uses env i's episode transform).
            sensor: Which sensor's episode-level state to use.

        Returns:
            Tensor with identical shape/dtype/device, values in [0, 1].
        """
        if not self.enabled:
            return img
        orig_shape = img.shape
        if img.dim() == 5:  # [B, T, C, H, W] -> fold time into batch
            B, T = img.shape[:2]
            img = img.reshape(B * T, *img.shape[2:])
            rep = T
        else:
            B, rep = img.shape[0], 1
        st = self._state[sensor]

        def _expand(x: torch.Tensor) -> torch.Tensor:
            return x.repeat_interleave(rep, dim=0) if rep > 1 else x

        out = img
        if self.use_ep_aug:
            # 1) Episode-level crop + resize back (camera shift/zoom).
            boxes = _expand(st["crop"])
            batch_idx = torch.arange(out.shape[0], device=out.device, dtype=out.dtype).unsqueeze(1)
            rois = torch.cat([batch_idx, boxes.to(out.dtype)], dim=1)
            out = roi_align(out, rois, output_size=self.img_hw, aligned=True)
            # 2) Episode-level color jitter.
            b, c, s, h = self.ep_bcsh
            out = _apply_color_jitter(
                out,
                _expand(st["b"]) if b > 0 else None,
                _expand(st["c"]) if c > 0 else None,
                _expand(st["s"]) if s > 0 else None,
                _expand(st["h"]) if h > 0 else None,
            )
        if self.randomize_color_channel:
            perm = _expand(st["perm"])
            out = out[torch.arange(out.shape[0], device=out.device).unsqueeze(1), perm]
        if self.use_t_aug:
            # Timestep-level jitter: resampled on every call.
            b, c, s, h = self.t_bcsh
            n = out.shape[0]
            out = _apply_color_jitter(
                out,
                self._sample_factor(b, n),
                self._sample_factor(c, n),
                self._sample_factor(s, n),
                self._sample_hue(h, n),
            )
        return out.clamp(0.0, 1.0).reshape(orig_shape)
