#!/usr/bin/env python3
"""
Visualise step 1 of the pipeline (motion estimation) for the report.

For each chosen frame k it runs the same corner detection, optical flow and
RANSAC fit as stabilize.py on the pair (k, k+1) and saves:
    pair_<k>_corners.png   frame k with the detected Shi-Tomasi corners
    pair_<k>_tracked.png   frame k+1 with each corner's motion drawn as an arrow
                           (green = RANSAC inlier, red = rejected outlier)
    pair_<k>_figure.png    both side by side, titled with the fitted motion

Motion between consecutive frames is only a few pixels, so arrows are drawn
longer than the real displacement (automatic per pair, or --arrow-scale);
the figure title says by how much.

Usage:
    python3 visualize_tracking.py --video data/synthetic_shaky.mp4 \
                                  --gt data/synthetic_shaky_groundtruth.npz \
                                  --frames 50 140
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from stabilize import frames, gt_trajectories

# Same settings as estimate_motion() in stabilize.py
FEATURE_PARAMS = dict(maxCorners=400, qualityLevel=0.01, minDistance=15, blockSize=3)
RANSAC_THRESH = 3.0

# BGR for OpenCV drawing
CORNER = (0, 220, 255)
INLIER = (60, 200, 60)
OUTLIER = (50, 50, 230)


def bgr_to_mpl(c):
    return tuple(v / 255 for v in c[::-1])


def load_frames(path, ks, max_width):
    """Return {index: frame} for every k and k+1 requested."""
    want = set(ks) | {k + 1 for k in ks}
    got = {}
    for i, f in enumerate(frames(path, max_width)):
        if i in want:
            got[i] = f
        if i >= max(want):
            break
    short = [k for k in ks if k not in got or k + 1 not in got]
    if short:
        raise SystemExit(f"Video is too short for frame(s) {short} (need frame k and k+1).")
    return got


def track(f0, f1):
    """Corners in f0, their LK positions in f1, RANSAC inlier mask and fitted (dx, dy, dtheta)."""
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(g0, **FEATURE_PARAMS)
    if corners is None or len(corners) < 10:
        raise SystemExit("Too few corners found in this frame.")
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, corners, None)
    good = st.ravel() == 1
    p0, p1 = corners[good], nxt[good]
    m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_THRESH)
    if m is None:
        raise SystemExit("RANSAC could not fit a transform for this pair.")
    motion = (m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0]))
    return (corners.reshape(-1, 2), p0.reshape(-1, 2), p1.reshape(-1, 2),
            inl.ravel().astype(bool), motion)


def auto_scale(shift, width):
    """Whole-number arrow magnification that makes the median arrow ~25 px (at 640 px width)."""
    target = 25.0 * width / 640
    return float(np.clip(round(target / max(np.median(shift), 1e-3)), 1, 30))


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_corners(f, corners, thick):
    out = f.copy()
    for c in corners:
        cv2.circle(out, _pt(c), 3 + thick, CORNER, thick, cv2.LINE_AA)
    return out


def draw_tracks(f, p0, p1, inl, scale, thick):
    """Arrow from each corner's old position towards its new one, lengthened by `scale`."""
    out = f.copy()
    order = np.argsort(~inl)                   # inliers first so outliers are drawn on top
    for i in order:
        a, b = p0[i], p1[i]
        tip = a + scale * (b - a)
        color = INLIER if inl[i] else OUTLIER
        cv2.circle(out, _pt(a), 1 + thick, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(out, _pt(a), _pt(tip), color, thick, cv2.LINE_AA, tipLength=0.3)
    return out


def save_figure(img_a, img_b, k, n_corners, inl, shift, motion, true_motion, scale, out_png):
    # Explicit layout in inches: text block, panel titles, two images, legend
    h, w = img_a.shape[:2]
    pw, gap, side = 6.3, 0.25, 0.15
    ph = pw * h / w
    head = 0.75 if true_motion is not None else 0.5
    title, foot = 0.35, 0.45
    W, H = 2 * pw + gap + 2 * side, head + title + ph + foot
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([side / W, foot / H, pw / W, ph / H]),
            fig.add_axes([(side + pw + gap) / W, foot / H, pw / W, ph / H])]
    axes[0].imshow(img_a[:, :, ::-1])
    axes[0].set_title(f"Frame {k}: {n_corners} corners detected", fontsize=11, loc="left")
    axes[1].imshow(img_b[:, :, ::-1])
    axes[1].set_title(f"Frame {k + 1}: tracked motion (arrows ×{scale:g})", fontsize=11, loc="left")
    for ax in axes:
        ax.set_axis_off()

    dx, dy, da = motion
    text = (f"Estimated motion  Δx = {dx:+.2f} px,  Δy = {dy:+.2f} px,  Δθ = {np.degrees(da):+.3f}°"
            f"    |    median corner shift {np.median(shift):.2f} px,  "
            f"{inl.sum()}/{len(inl)} inliers")
    if true_motion is not None:
        tx, ty, ta = true_motion
        text += f"\nTrue motion           Δx = {tx:+.2f} px,  Δy = {ty:+.2f} px,  Δθ = {np.degrees(ta):+.3f}°"
    fig.text(side / W, 1 - 0.12 / H, text, fontsize=10, ha="left", va="top", family="monospace")

    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec=bgr_to_mpl(CORNER), label="detected corner"),
               Line2D([], [], color=bgr_to_mpl(INLIER), label="tracked, RANSAC inlier"),
               Line2D([], [], color=bgr_to_mpl(OUTLIER), label="tracked, RANSAC outlier")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02 / H), ncol=3,
               fontsize=9, frameon=False)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", help="ground-truth .npz from synth_video.py (optional, adds true motion)")
    ap.add_argument("--frames", type=int, nargs="+", default=[50, 140],
                    help="frame indices k; each is paired with k+1 (default: 50 140)")
    ap.add_argument("--arrow-scale", type=float, default=None,
                    help="draw arrows this many times the real displacement "
                         "(default: chosen per pair so the median arrow is ~25 px)")
    ap.add_argument("--max-width", type=int, default=960, help="same downscaling as stabilize.py")
    ap.add_argument("--out", default="results_tracking")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    imgs = load_frames(args.video, args.frames, args.max_width)
    gt_shaky = gt_trajectories(args.gt)[0] if args.gt else None

    for k in args.frames:
        f0, f1 = imgs[k], imgs[k + 1]
        thick = max(1, int(round(f0.shape[1] / 640)))
        corners, p0, p1, inl, motion = track(f0, f1)
        shift = np.linalg.norm(p1 - p0, axis=1)
        scale = args.arrow_scale or auto_scale(shift, f0.shape[1])
        true_motion = None
        if gt_shaky is not None and k + 1 < len(gt_shaky):
            true_motion = gt_shaky[k + 1] - gt_shaky[k]

        img_a = draw_corners(f0, corners, thick)
        img_b = draw_tracks(f1, p0, p1, inl, scale, thick)
        base = os.path.join(args.out, f"pair_{k}")
        cv2.imwrite(base + "_corners.png", img_a)
        cv2.imwrite(base + "_tracked.png", img_b)
        save_figure(img_a, img_b, k, len(corners), inl, shift, motion, true_motion,
                    scale, base + "_figure.png")

        dx, dy, da = motion
        line = (f"frames {k}->{k + 1}: {len(corners)} corners, {inl.sum()}/{len(inl)} inliers, "
                f"arrows x{scale:g}, "
                f"dx={dx:+.2f} dy={dy:+.2f} dtheta={np.degrees(da):+.3f} deg")
        if true_motion is not None:
            line += (f"  (true dx={true_motion[0]:+.2f} dy={true_motion[1]:+.2f} "
                     f"dtheta={np.degrees(true_motion[2]):+.3f} deg)")
        print(line)
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
