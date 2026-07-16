#!/usr/bin/env python
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test + visualization for TacSL-style tactile image augmentation.

Runs WITHOUT Isaac Sim (pure torch): builds a batch of synthetic tactile images
from the real GelSight background (bg.jpg) plus fake contact blobs, applies
episode-level augmentation only (``use_t_aug=False``; default ``ep_bcsh`` + crop
from :class:`TactileImageAugmentor`), saves a before/after grid, and verifies:

  1. output shape/dtype/range match the input contract ([0, 1] floats)
  2. episode-level transform is FIXED across timesteps
  3. reset() changes the episode-level transform

Note: the RL env still uses the augmentor constructor defaults (ep + t). This
demo intentionally disables timestep aug for clearer visualization.

Usage:
    python scripts/demos/factory/test_tactile_augmentation.py [--out tactile_aug_test.png] [--device cuda]
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AUG_PATH = os.path.join(
    REPO_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "direct", "factory", "tactile_augmentation.py"
)
BG_PATH = os.path.join(
    REPO_ROOT, "source", "isaaclab", "isaaclab", "sensors", "tacsl_sensor", "gelsight_r15_data", "bg.jpg"
)

# Import the augmentor directly from its file to avoid pulling in Isaac Sim
# through the isaaclab_tasks package __init__.
spec = importlib.util.spec_from_file_location("tactile_augmentation", AUG_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TactileImageAugmentor = mod.TactileImageAugmentor


def make_batch(num_envs: int, hw: tuple[int, int], device: str) -> torch.Tensor:
    """Synthetic tactile batch: real background + one fake circular indentation per env."""
    H, W = hw
    bg = cv2.cvtColor(cv2.imread(BG_PATH), cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg, (W, H)).astype(np.float32) / 255.0
    imgs = np.repeat(bg[None], num_envs, axis=0)
    rng = np.random.default_rng(0)
    for i in range(num_envs):
        cx, cy = rng.integers(W // 4, 3 * W // 4), rng.integers(H // 4, 3 * H // 4)
        rr = int(min(H, W) * 0.2)
        yy, xx = np.mgrid[0:H, 0:W]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < rr**2
        # brighten a blob to mimic a contact imprint
        imgs[i][mask] = np.clip(imgs[i][mask] * 1.6 + 0.1, 0, 1)
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)  # [B, 3, H, W]


def save_grid(rows: list[torch.Tensor], labels: list[str], path: str, upscale: int = 3) -> None:
    """rows: list of [B, 3, H, W] tensors -> stacked labeled grid PNG."""
    panels = []
    for row, label in zip(rows, labels):
        imgs = (row.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        strip = np.concatenate(list(imgs), axis=1)
        strip = cv2.resize(strip, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
        strip = cv2.copyMakeBorder(strip, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(strip, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(strip)
    grid = np.concatenate(panels, axis=0)
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="tactile_aug_test.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    args = parser.parse_args()

    hw = (80, 60)
    batch = make_batch(args.num_envs, hw, args.device)

    # Demo-only: disable t_aug. Default ep_bcsh/crop come from TactileImageAugmentor.
    # RL env keeps constructor defaults (use_t_aug=True).
    aug = TactileImageAugmentor(
        num_envs=args.num_envs,
        device=args.device,
        img_hw=hw,
        use_t_aug=False,
    )

    # --- 1. shape / dtype / range ---
    out = aug.apply(batch)
    assert out.shape == batch.shape, f"shape changed: {batch.shape} -> {out.shape}"
    assert out.dtype == batch.dtype, f"dtype changed: {batch.dtype} -> {out.dtype}"
    assert out.min() >= 0.0 and out.max() <= 1.0, f"range violated: [{out.min()}, {out.max()}]"
    print(
        f"[PASS] shape/dtype preserved, values in [0, 1] "
        f"(use_t_aug={aug.use_t_aug}, ep_bcsh={aug.ep_bcsh}, crop_scale={aug.crop_scale})"
    )

    # --- 2. episode-level transform fixed across timesteps ---
    step_a = aug.apply(batch)
    step_b = aug.apply(batch)
    assert torch.allclose(step_a, step_b, atol=1e-6), "episode transform changed between timesteps!"
    print("[PASS] episode-level transform constant across timesteps")

    # --- 3. reset() resamples the episode transform ---
    crop_before = aug._state["left"]["crop"].clone()
    hue_before = aug._state["left"]["h"].clone()
    aug.reset(torch.arange(args.num_envs, device=args.device))
    crop_changed = not torch.allclose(crop_before, aug._state["left"]["crop"])
    hue_changed = not torch.allclose(hue_before, aug._state["left"]["h"])
    assert crop_changed or hue_changed, "reset() did not change the episode transform!"
    step_c = aug.apply(batch)
    assert not torch.allclose(step_a, step_c, atol=1e-4), "output unchanged after reset!"
    print("[PASS] reset() resamples the episode-level transform")

    # --- 3b. partial reset only affects the given env ids ---
    step_d_ref = aug.apply(batch)
    aug.reset(torch.tensor([0], device=args.device))
    step_d = aug.apply(batch)
    assert torch.allclose(step_d_ref[1:], step_d[1:], atol=1e-6), "partial reset touched other envs!"
    print("[PASS] partial reset(env_ids) leaves other envs' transforms untouched")

    # --- visualization grid ---
    save_grid(
        [batch, step_a, step_c, aug.apply(batch)],
        [
            "original",
            "ep aug (episode A)",
            "ep aug (episode B, after reset)",
            "ep aug (same as B, call 2)",
        ],
        args.out,
    )
    print(f"[DONE] all checks passed — visualization saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
