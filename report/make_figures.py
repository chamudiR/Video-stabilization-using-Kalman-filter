#!/usr/bin/env python3
"""
Collect and generate every figure used by report.tex into report/figures/.

Run stabilize.py on both videos first (results_synthetic/ and results_real/),
then, from the project folder:
    python3 report/make_figures.py
"""

import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")

RUNS = {
    "synth": dict(results="results_synthetic", video="data/synthetic_shaky.mp4",
                  gt="data/synthetic_shaky_groundtruth.npz", track=[50, 140], mean_start=40),
    "real": dict(results="results_real", video="data/videoShake.m4v",
                 gt=None, track=[250, 112], mean_start=60),
}
PLOTS = ["trajectories.png", "trajectories_pan_zoom.png", "tradeoff_q_sweep.png", "kf_diagnostics.png"]


def temporal_mean(video, start, n, out_png):
    """Average n consecutive frames of the side-by-side video: residual shake shows up as blur."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    acc = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        acc.append(f.astype(float))
    cap.release()
    if not acc:
        raise SystemExit(f"Could not read frames from {video}")
    cv2.imwrite(out_png, np.mean(acc, axis=0).astype(np.uint8))


def main():
    os.makedirs(FIG, exist_ok=True)
    for tag, run in RUNS.items():
        res = os.path.join(ROOT, run["results"])
        if not os.path.isdir(res):
            raise SystemExit(f"Missing {res}: run stabilize.py on {run['video']} first.")
        for p in PLOTS:
            shutil.copy(os.path.join(res, p), os.path.join(FIG, f"{tag}_{p}"))

        tmp = os.path.join(FIG, f"_{tag}_tracking")
        cmd = [sys.executable, os.path.join(ROOT, "visualize_tracking.py"),
               "--video", os.path.join(ROOT, run["video"]), "--out", tmp,
               "--frames", *map(str, run["track"])]
        if run["gt"]:
            cmd += ["--gt", os.path.join(ROOT, run["gt"])]
        subprocess.run(cmd, check=True)
        for k in run["track"]:
            shutil.copy(os.path.join(tmp, f"pair_{k}_figure.png"),
                        os.path.join(FIG, f"{tag}_tracking_{k}.png"))
        shutil.rmtree(tmp)

        temporal_mean(os.path.join(res, "comparison_raw_kf_rts.mp4"), run["mean_start"], 10,
                      os.path.join(FIG, f"{tag}_temporal_mean.png"))
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
