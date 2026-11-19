#!/usr/bin/env python3
"""
Generate a synthetic shaky video with known ground truth.

A virtual camera looks at a large textured scene. The camera follows an
"intended" path (slow pan, then a fast deliberate pan, then slow again) and
on top of that we add hand shake (band-limited random translation and
rotation). Because we know both the shaky and the intended camera motion
exactly, we can measure how close each filter gets to the intended path.

Outputs:
    <out>.mp4                    the shaky video
    <out>_groundtruth.npz        per-frame 2x3 affine matrices (shaky and intended)

Usage:
    python synth_video.py --out data/synthetic_shaky
"""

import argparse
import os

import cv2
import numpy as np


def make_scene(width, height, rng):
    """Random textured scene with plenty of corners for feature tracking."""
    scene = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (0, 0), 6)
    for _ in range(450):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        kind = rng.integers(0, 3)
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        if kind == 0:
            w, h = int(rng.integers(10, 80)), int(rng.integers(10, 80))
            cv2.rectangle(scene, (x, y), (x + w, y + h), color, -1)
        elif kind == 1:
            cv2.circle(scene, (x, y), int(rng.integers(5, 40)), color, -1)
        else:
            x2, y2 = x + int(rng.integers(-100, 100)), y + int(rng.integers(-100, 100))
            cv2.line(scene, (x, y), (x2, y2), color, int(rng.integers(1, 5)))
    return scene


def band_limited_noise(n, std, smooth_frames, rng):
    """Gaussian noise low-pass filtered so it looks like hand tremor, not white noise."""
    raw = rng.normal(0, 1, n + 6 * smooth_frames)
    k = cv2.getGaussianKernel(6 * smooth_frames + 1, smooth_frames).ravel()
    filt = np.convolve(raw, k, mode="same")[3 * smooth_frames: 3 * smooth_frames + n]
    return filt / (filt.std() + 1e-12) * std


def intended_path(n):
    """Slow pan right, a fast deliberate pan, then slow again; gentle vertical drift."""
    vx = np.full(n, 1.0)
    a, b = int(0.40 * n), int(0.55 * n)        # fast pan window
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, b - a))
    vx[a:b] += 7.0 * ramp
    cam_x = np.concatenate([[0.0], np.cumsum(vx[1:])])
    cam_y = 15.0 * np.sin(np.linspace(0, 1.5 * np.pi, n))
    return cam_x, cam_y


def frame_matrix(cam_x, cam_y, theta, w, h, x0, y0):
    """2x3 affine mapping scene coordinates -> frame coordinates."""
    T = np.array([[1, 0, -(x0 + cam_x)],
                  [0, 1, -(y0 + cam_y)],
                  [0, 0, 1]], dtype=float)
    Rc = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(theta), 1.0)
    R = np.vstack([Rc, [0, 0, 1]])
    return (R @ T)[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic_shaky")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--shake-px", type=float, default=6.0, help="std of translational shake (px)")
    ap.add_argument("--shake-deg", type=float, default=0.6, help="std of rotational shake (deg)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, w, h = args.frames, args.width, args.height

    cam_x, cam_y = intended_path(n)
    margin = 120
    scene = make_scene(int(w + cam_x.max() + 2 * margin), int(h + 2 * margin + 40), rng)
    x0, y0 = margin, margin + 20

    shake_x = band_limited_noise(n, args.shake_px, 2, rng)
    shake_y = band_limited_noise(n, args.shake_px, 2, rng)
    shake_t = band_limited_noise(n, np.radians(args.shake_deg), 2, rng)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    M_shaky, M_intended = [], []
    for k in range(n):
        Ms = frame_matrix(cam_x[k] + shake_x[k], cam_y[k] + shake_y[k], shake_t[k], w, h, x0, y0)
        Mi = frame_matrix(cam_x[k], cam_y[k], 0.0, w, h, x0, y0)
        frame = cv2.warpAffine(scene, Ms, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 2.0, frame.shape)                 # mild sensor noise
        frame = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        M_shaky.append(Ms)
        M_intended.append(Mi)
    writer.release()

    np.savez(args.out + "_groundtruth.npz",
             M_shaky=np.array(M_shaky), M_intended=np.array(M_intended), fps=args.fps)
    print(f"Wrote {args.out}.mp4 ({n} frames) and {args.out}_groundtruth.npz")


if __name__ == "__main__":
    main()
