#!/usr/bin/env python3
"""
Video stabilisation with a linear Kalman filter and RTS smoother.

Pipeline
    1. Motion estimation: track corners between consecutive frames (Shi-Tomasi
       + pyramidal Lucas-Kanade) and fit a similarity transform with RANSAC,
       giving per-frame (dx, dy, dtheta).
    2. Trajectory: cumulative sum of the per-frame motion = raw camera path.
    3. Smoothing: estimate the *intended* camera path with
         - causal moving average   (baseline, past frames only)
         - Kalman filter           (causal, constant-velocity model)
         - centred moving average  (baseline, past + future frames)
         - RTS smoother            (Kalman forward pass + backward pass)
    4. Correction: warp each frame by (smoothed - raw) so the content follows
       the smoothed path; zoom slightly to hide the moving borders.
    5. Evaluation: inter-frame PSNR (ITF), trajectory jitter, crop required,
       and (for synthetic video) RMSE against the true intended path.
       Plus a Q sweep for the smoothness-vs-lag tradeoff and NIS consistency.

Usage
    python stabilize.py --video data/synthetic_shaky.mp4 \
                        --gt data/synthetic_shaky_groundtruth.npz
    python stabilize.py --video my_phone_clip.mp4
"""

import argparse
import json
import os
import subprocess
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kalman import cv_model, kalman_filter, rts_smoother, moving_average

AXES = ["x", "y", "theta"]
AXIS_LABELS = {"x": "x translation (px)", "y": "y translation (px)", "theta": "rotation (deg)"}

# Plot styling: fixed categorical order (never cycled); raw and ground truth are
# neutral references; dashed lines give the baselines a non-colour cue.
STYLE = {
    "raw":         dict(color="#9a9a94", lw=1.0, ls="-",  label="Raw (shaky)"),
    "gt":          dict(color="#1f1f1d", lw=1.2, ls=":",  label="True intended path"),
    "kf":          dict(color="#2a78d6", lw=2.0, ls="-",  label="Kalman filter (causal)"),
    "rts":         dict(color="#eb6834", lw=2.0, ls="-",  label="RTS smoother (offline)"),
    "ma_causal":   dict(color="#1baf7a", lw=1.6, ls="--", label="Moving avg, causal"),
    "ma_centred":  dict(color="#c98500", lw=1.6, ls="--", label="Moving avg, centred"),
}
METHODS = ["ma_causal", "kf", "ma_centred", "rts"]


# --------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------
def frames(path, max_width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_width and f.shape[1] > max_width:
            s = max_width / f.shape[1]
            f = cv2.resize(f, (max_width, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA)
        yield f
    cap.release()


def video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


# --------------------------------------------------------------------------
# 1. Motion estimation
# --------------------------------------------------------------------------

def estimate_motion(path, max_width):
    """Per-frame similarity motion (dx, dy, dtheta) between consecutive frames."""
    transforms, inlier_ratio = [], []
    prev_gray = None
    for f in frames(path, max_width):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                          minDistance=15, blockSize=3)
            m = None
            if pts is not None and len(pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                good = st.ravel() == 1
                if good.sum() >= 10:
                    m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good],
                                                         method=cv2.RANSAC,
                                                         ransacReprojThreshold=3.0)
                    if inl is not None:
                        inlier_ratio.append(float(inl.mean()))
            if m is None:                      # tracking failed: assume no motion
                transforms.append((0.0, 0.0, 0.0))
            else:
                transforms.append((m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])))
        prev_gray = gray
    if not transforms:
        raise SystemExit("Video has fewer than two frames.")
    return np.array(transforms), (float(np.mean(inlier_ratio)) if inlier_ratio else 0.0)


def trajectory(transforms):
    """Cumulative camera path, with frame 0 at the origin. Shape (N, 3)."""
    return np.vstack([np.zeros(3), np.cumsum(transforms, axis=0)])


def gt_trajectories(npz_path):
    """True shaky and intended paths in the same parametrisation as the estimator."""
    d = np.load(npz_path)

    def path(Ms):
        rel = []
        for k in range(1, len(Ms)):
            A = np.vstack([Ms[k], [0, 0, 1]])
            B = np.vstack([Ms[k - 1], [0, 0, 1]])
            R = A @ np.linalg.inv(B)            # frame k-1 coords -> frame k coords
            rel.append((R[0, 2], R[1, 2], np.arctan2(R[1, 0], R[0, 0])))
        return trajectory(np.array(rel))

    return path(d["M_shaky"]), path(d["M_intended"])


# --------------------------------------------------------------------------
# 2. Smoothing
# --------------------------------------------------------------------------
def estimate_R(traj, window=15):
    """
    Data-driven measurement-noise variance per axis: variance of the
    high-frequency residual (trajectory minus a centred moving average).
    This is a heuristic; the shake is coloured noise, not white.
    """
    R = []
    for a in range(traj.shape[1]):
        resid = traj[:, a] - moving_average(traj[:, a], window)
        R.append(max(np.var(resid), 1e-10))
    return np.array(R)


def smooth(traj, q_ratio, R, ma_window):
    """Apply all four smoothers to every axis. Returns paths and KF diagnostics."""
    out = {m: np.zeros_like(traj) for m in METHODS}
    diag = {}
    for a, name in enumerate(AXES):
        z = traj[:, a]
        F, H, Q, Rm = cv_model(q=q_ratio * R[a], r=R[a])
        kf = kalman_filter(z, F, H, Q, Rm)
        rts = rts_smoother(kf, F)
        out["kf"][:, a] = kf["x_filt"][:, 0]
        out["rts"][:, a] = rts["x_smooth"][:, 0]
        out["ma_causal"][:, a] = moving_average(z, ma_window, causal=True)
        out["ma_centred"][:, a] = moving_average(z, ma_window, causal=False)
        diag[name] = {"kf": kf, "rts": rts}
    return out, diag


# --------------------------------------------------------------------------
# 3. Correction + rendering
# --------------------------------------------------------------------------
def correction_matrix(corr, zoom, w, h):
    """Warp that moves frame content by corr = (dx, dy, dtheta), then zooms about the centre."""
    dx, dy, da = corr
    c, s = np.cos(da), np.sin(da)
    M = np.array([[c, -s, dx], [s, c, dy], [0, 0, 1.0]])
    Z = np.array([[zoom, 0, (1 - zoom) * w / 2],
                  [0, zoom, (1 - zoom) * h / 2],
                  [0, 0, 1.0]])
    return (Z @ M)[:2]


def central(gray, frac=0.8):
    h, w = gray.shape
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return gray[mh:h - mh, mw:w - mw]


def render_and_score(path, raw_traj, paths, zoom, max_width, out_video, fps, show=("kf", "rts")):
    """
    Warp every frame for every method, compute ITF (mean inter-frame PSNR),
    and write a side-by-side comparison video: raw | KF | RTS.
    """
    names = ["raw"] + METHODS
    corrections = {"raw": np.zeros_like(raw_traj)}
    for m in METHODS:
        corrections[m] = paths[m] - raw_traj

    psnr = {n: [] for n in names}
    prev = {}
    writer = None
    for k, f in enumerate(frames(path, max_width)):
        h, w = f.shape[:2]
        panels = {}
        for n in names:
            M = correction_matrix(corrections[n][k], 1.0 if n == "raw" else zoom, w, h)
            g = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            panels[n] = g
            gray = central(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY))
            if n in prev:
                psnr[n].append(cv2.PSNR(prev[n], gray))
            prev[n] = gray

        tiles = []
        for n in ("raw",) + tuple(show):
            tw = min(w, 640)                       # keep the 3-panel video a sensible size
            t = cv2.resize(panels[n], (tw, int(round(h * tw / w)))) if tw < w else panels[n].copy()
            label = STYLE[n]["label"]
            cv2.rectangle(t, (0, 0), (t.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(t, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
        combo = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
        if writer is None:
            writer = cv2.VideoWriter(out_video + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
    writer.release()
    _to_h264(out_video + ".tmp.mp4", out_video)
    return {n: float(np.mean(v)) for n, v in psnr.items()}


def _to_h264(src, dst):
    """Re-encode to H.264 so the video plays in browsers/PowerPoint (falls back to mp4v)."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "22", dst])
        if r.returncode == 0:
            os.remove(src)
            return
    os.replace(src, dst)


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------
def jitter(p):
    """RMS of the second difference of x, y (px/frame^2): lower = smoother path."""
    d2 = np.diff(p[:, :2], n=2, axis=0)
    return float(np.sqrt(np.mean(d2 ** 2)))


def rmse_xy(p, ref, align=True):
    """
    RMSE of the x, y path against a reference. With align=True the best constant
    offset is removed first: the stabilised video's absolute position is arbitrary
    (frame 0 itself contains shake), so only the shape of the path matters.
    """
    e = p[:, :2] - ref[:, :2]
    if align:
        e = e - e.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def max_correction(p, raw):
    return float(np.max(np.abs(p[:, :2] - raw[:, :2])))


def nis_stats(kf):
    """Normalised innovation squared; for a consistent 1-D filter E[NIS] = 1."""
    nis = kf["innov"] ** 2 / kf["S"]
    return float(np.mean(nis[5:])), nis


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
def _style_axes(ax):
    ax.grid(True, color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b4b2a9")
    ax.tick_params(colors="#5f5e5a", labelsize=9)


def _plot_line(ax, x, y, key, **kw):
    st = dict(STYLE[key]); st.update(kw)
    ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])


def plot_trajectories(raw, paths, gt_int, out_png, zoom_range=None):
    if gt_int is not None:
        # absolute offset is arbitrary (frame 0 is itself shaken): align by best constant
        gt_int = gt_int + (raw - gt_int).mean(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    t = np.arange(len(raw))
    for a, name in enumerate(AXES):
        ax = axes[a]
        conv = np.degrees if name == "theta" else (lambda v: v)
        _plot_line(ax, t, conv(raw[:, a]), "raw")
        for m in METHODS:
            _plot_line(ax, t, conv(paths[m][:, a]), m)
        if gt_int is not None:
            _plot_line(ax, t, conv(gt_int[:, a]), "gt")
        ax.set_ylabel(AXIS_LABELS[name], fontsize=10, color="#3d3d3a")
        _style_axes(ax)
    axes[-1].set_xlabel("frame", fontsize=10, color="#3d3d3a")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Camera trajectory: raw vs smoothed", fontsize=12, x=0.01, ha="left", color="#1f1f1d")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Zoomed view of x around the fast pan, where lag differences show up
    if zoom_range:
        a0, a1 = zoom_range
        fig, ax = plt.subplots(figsize=(10, 4.8))
        tt = t[a0:a1]
        _plot_line(ax, tt, raw[a0:a1, 0], "raw")
        for m in METHODS:
            _plot_line(ax, tt, paths[m][a0:a1, 0], m)
        if gt_int is not None:
            _plot_line(ax, tt, gt_int[a0:a1, 0], "gt")
        ax.set_xlabel("frame", fontsize=10, color="#3d3d3a")
        ax.set_ylabel(AXIS_LABELS["x"], fontsize=10, color="#3d3d3a")
        fig.suptitle("Zoom on the deliberate pan: causal methods lag, offline methods do not",
                     fontsize=11, x=0.01, ha="left", color="#1f1f1d")
        _style_axes(ax)
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.84])
        fig.savefig(out_png.replace(".png", "_pan_zoom.png"), dpi=150)
        plt.close(fig)


def plot_tradeoff(sweep, out_png, y_label):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    markers = {"ma_causal": "s", "kf": "o", "ma_centred": "D", "rts": "^"}
    for m in METHODS:
        pts = sweep[m]
        st = STYLE[m]
        ax.plot([p["jitter"] for p in pts], [p["y"] for p in pts], color=st["color"], ls=st["ls"],
                lw=1.4, marker=markers[m], ms=6, label=st["label"])
    ax.set_xlabel("jitter: RMS 2nd difference of path (px/frame²)  ← smoother", fontsize=10, color="#3d3d3a")
    ax.set_ylabel(y_label, fontsize=10, color="#3d3d3a")
    ax.set_xscale("log")
    ax.set_title("Smoothness vs fidelity tradeoff (sweeping Q or window)", fontsize=11,
                 loc="left", color="#1f1f1d")
    _style_axes(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_kf_diagnostics(diag_x, out_png):
    kf = diag_x["kf"]
    mean_nis, nis = nis_stats(kf)
    t = np.arange(len(kf["innov"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    bound = 2 * np.sqrt(kf["S"])
    ax.fill_between(t, -bound, bound, color="#cde2fb", label="±2√S (95% expected)")
    ax.plot(t, kf["innov"], color="#2a78d6", lw=1.0, label="innovation z − Hx̂⁻")
    inside = np.mean(np.abs(kf["innov"]) <= bound) * 100
    ax.set_ylabel("innovation (px)", fontsize=10, color="#3d3d3a")
    ax.set_title(f"x-axis innovations: {inside:.0f}% inside ±2√S", fontsize=11, loc="left", color="#1f1f1d")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style_axes(ax)

    ax = axes[1]
    ax.plot(t, nis, color="#2a78d6", lw=1.0, label="NIS")
    ax.axhline(1.0, color="#1f1f1d", ls=":", lw=1.2, label="expected value (1)")
    ax.axhline(3.84, color="#e34948", ls="--", lw=1.0, label="χ²₁ 95% bound (3.84)")
    ax.set_ylim(1e-3, None)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=3)
    ax.set_ylabel("NIS", fontsize=10, color="#3d3d3a")
    ax.set_yscale("log")
    ax.set_title(f"Normalised innovation squared (mean {mean_nis:.2f})", fontsize=11, loc="left", color="#1f1f1d")
    _style_axes(ax)

    ax = axes[2]
    ax.plot(t, kf["K"][:, 0], color="#2a78d6", lw=1.6, label="K (position)")
    ax.plot(t, kf["K"][:, 1], color="#eb6834", lw=1.6, ls="--", label="K (velocity)")
    ax.set_ylabel("Kalman gain", fontsize=10, color="#3d3d3a")
    ax.set_xlabel("frame", fontsize=10, color="#3d3d3a")
    ax.set_title("Gain converges to a steady-state value", fontsize=11, loc="left", color="#1f1f1d")
    ax.legend(fontsize=8, frameon=False)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return mean_nis, inside


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", help="ground-truth .npz from synth_video.py (optional)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--q-ratio", type=float, default=3e-4,
                    help="process-to-measurement noise ratio Q/R (smaller = smoother, more lag)")
    ap.add_argument("--ma-window", type=int, default=15, help="moving-average window (frames)")
    ap.add_argument("--zoom", type=float, default=1.12, help="zoom to hide moving borders")
    ap.add_argument("--max-width", type=int, default=960, help="downscale wider videos for speed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fps = video_fps(args.video)

    print("1/5 Estimating inter-frame motion ...")
    transforms, inlier = estimate_motion(args.video, args.max_width)
    raw = trajectory(transforms)
    n = len(raw)
    print(f"    {n} frames, mean RANSAC inlier ratio {inlier:.2f}")

    gt_shaky = gt_int = None
    if args.gt:
        gt_shaky, gt_int = gt_trajectories(args.gt)
        est_err = rmse_xy(raw, gt_shaky, align=False)
        print(f"    motion-estimation check: raw path vs true shaky path RMSE = {est_err:.2f} px")

    print("2/5 Smoothing (MA causal, KF, MA centred, RTS) ...")
    R = estimate_R(raw)
    paths, diag = smooth(raw, args.q_ratio, R, args.ma_window)

    print("3/5 Sweeping Q and window for the tradeoff curve ...")
    sweep = {m: [] for m in METHODS}
    ref = gt_int
    for q in np.logspace(-6, 0, 13):
        p, _ = smooth(raw, q, R, args.ma_window)
        for m in ("kf", "rts"):
            y = rmse_xy(p[m], ref) if ref is not None else max_correction(p[m], raw)
            sweep[m].append({"param": float(q), "jitter": jitter(p[m]), "y": y})
    for wdw in (3, 5, 9, 15, 21, 31, 45, 61, 91):
        for m, causal in (("ma_causal", True), ("ma_centred", False)):
            pm = np.column_stack([moving_average(raw[:, a], wdw, causal) for a in range(3)])
            y = rmse_xy(pm, ref) if ref is not None else max_correction(pm, raw)
            sweep[m].append({"param": wdw, "jitter": jitter(pm), "y": y})
    y_label = ("RMSE to true intended path (px)  ← more faithful" if ref is not None
               else "max correction needed (px)  ← less crop")
    plot_tradeoff(sweep, os.path.join(args.out, "tradeoff_q_sweep.png"), y_label)

    print("4/5 Rendering stabilised frames and scoring ...")
    itf = render_and_score(args.video, raw, paths, args.zoom, args.max_width,
                           os.path.join(args.out, "comparison_raw_kf_rts.mp4"), fps)

    print("5/5 Plots and metrics ...")
    pan = None
    if ref is not None:
        speed = np.abs(np.gradient(ref[:, 0]))
        hi = np.where(speed > 2 * np.median(speed))[0]
        if len(hi):
            pan = (max(0, hi[0] - 30), min(n, hi[-1] + 30))
    else:
        speed = np.abs(np.gradient(paths["rts"][:, 0]))
        c = int(np.argmax(speed))
        pan = (max(0, c - 60), min(n, c + 60))
    plot_trajectories(raw, paths, gt_int, os.path.join(args.out, "trajectories.png"), pan)
    mean_nis, inside = plot_kf_diagnostics(diag["x"], os.path.join(args.out, "kf_diagnostics.png"))

    metrics = {"settings": {"q_ratio": args.q_ratio, "ma_window": args.ma_window, "zoom": args.zoom,
                            "R_estimated": {a: float(R[i]) for i, a in enumerate(AXES)},
                            "frames": n, "fps": fps, "ransac_inlier_ratio": inlier},
               "methods": {}}
    for name in ["raw"] + METHODS:
        p = raw if name == "raw" else paths[name]
        row = {"ITF_psnr_dB": itf[name], "jitter_px": jitter(p),
               "max_correction_px": max_correction(p, raw)}
        if ref is not None:
            row["rmse_to_intended_px"] = rmse_xy(p, ref)
        metrics["methods"][name] = row
    metrics["kf_consistency_x"] = {"mean_NIS": mean_nis, "pct_innov_within_2sigma": inside}
    if gt_shaky is not None:
        metrics["motion_estimation_rmse_px"] = rmse_xy(raw, gt_shaky, align=False)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    hdr = f"{'method':<24}{'ITF (dB)':>10}{'jitter':>10}{'max corr':>10}"
    hdr += f"{'RMSE gt':>10}" if ref is not None else ""
    print("\n" + hdr)
    for name, row in metrics["methods"].items():
        line = f"{STYLE[name]['label']:<24}{row['ITF_psnr_dB']:>10.2f}{row['jitter_px']:>10.3f}{row['max_correction_px']:>10.1f}"
        if ref is not None:
            line += f"{row['rmse_to_intended_px']:>10.2f}"
        print(line)
    print(f"\nKF consistency (x): mean NIS {mean_nis:.2f}, {inside:.0f}% of innovations within ±2√S")
    print(f"Outputs in {args.out}/")


if __name__ == "__main__":
    main()
