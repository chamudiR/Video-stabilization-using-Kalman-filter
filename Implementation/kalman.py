"""
Linear Kalman filter, Rauch-Tung-Striebel smoother and moving-average baselines
for smoothing a 1-D camera trajectory (one call per axis: x, y, theta).

State model (constant velocity, dt = 1 frame):
    x_k = [p_k, v_k]^T
    x_k = F x_{k-1} + G a_k,      a_k ~ N(0, q)     (random acceleration)
    z_k = H x_k + v_k,            v_k ~ N(0, r)     (shake + estimation noise)

    F = [[1, dt],      G = [[dt^2 / 2],     H = [1, 0]
         [0,  1]]           [dt      ]]

    Q = q * G G^T  (discrete white-noise acceleration model)
"""

import numpy as np


def cv_model(q, r, dt=1.0):
    """Return F, H, Q, R for the constant-velocity model."""
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    G = np.array([[0.5 * dt ** 2],
                  [dt]])
    Q = q * (G @ G.T)
    R = np.array([[r]])
    return F, H, Q, R


def kalman_filter(z, F, H, Q, R, x0=None, P0=None):
    """
    Run a linear Kalman filter over measurements z (shape (N,)).

    Returns a dict with predicted and filtered states/covariances, gains,
    innovations and innovation covariances (needed for the RTS smoother
    and for consistency checks such as NIS).
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    nx = F.shape[0]
    I = np.eye(nx)

    if x0 is None:
        x0 = np.array([z[0], 0.0])
    if P0 is None:
        r = R[0, 0]
        P0 = np.diag([r, r])

    x_pred = np.zeros((n, nx))
    P_pred = np.zeros((n, nx, nx))
    x_filt = np.zeros((n, nx))
    P_filt = np.zeros((n, nx, nx))
    K_hist = np.zeros((n, nx))
    innov = np.zeros(n)
    S_hist = np.zeros(n)

    for k in range(n):
        # ---- Predict ----
        if k == 0:
            xp, Pp = x0.astype(float), P0.astype(float)
        else:
            xp = F @ x_filt[k - 1]
            Pp = F @ P_filt[k - 1] @ F.T + Q

        # ---- Update ----
        y = z[k] - (H @ xp)[0]                     # innovation
        S = (H @ Pp @ H.T + R)[0, 0]               # innovation covariance
        K = (Pp @ H.T)[:, 0] / S                   # Kalman gain
        xf = xp + K * y
        KH = np.outer(K, H[0])
        # Joseph form: numerically stable and keeps P symmetric positive definite
        Pf = (I - KH) @ Pp @ (I - KH).T + np.outer(K, K) * R[0, 0]

        x_pred[k], P_pred[k] = xp, Pp
        x_filt[k], P_filt[k] = xf, Pf
        K_hist[k], innov[k], S_hist[k] = K, y, S

    return {
        "x_pred": x_pred, "P_pred": P_pred,
        "x_filt": x_filt, "P_filt": P_filt,
        "K": K_hist, "innov": innov, "S": S_hist,
    }


def rts_smoother(kf, F):
    """
    Rauch-Tung-Striebel fixed-interval smoother.

    Runs backwards over the Kalman filter output so every estimate uses
    past AND future measurements. Possible here because stabilisation of a
    recorded video is an offline problem.
    """
    x_f, P_f = kf["x_filt"], kf["P_filt"]
    x_p, P_p = kf["x_pred"], kf["P_pred"]
    n = len(x_f)

    x_s = x_f.copy()
    P_s = P_f.copy()
    for k in range(n - 2, -1, -1):
        C = P_f[k] @ F.T @ np.linalg.inv(P_p[k + 1])   # smoother gain
        x_s[k] = x_f[k] + C @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_p[k + 1]) @ C.T
    return {"x_smooth": x_s, "P_smooth": P_s}


def moving_average(z, window, causal=False):
    """
    Moving-average baseline.
    causal=True  -> trailing window (uses only past frames, like the KF)
    causal=False -> centred window (uses past and future, like the RTS smoother)
    """
    z = np.asarray(z, dtype=float)
    window = max(1, int(window))
    if causal:
        padded = np.concatenate([np.full(window - 1, z[0]), z])
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")
    half = window // 2
    padded = np.pad(z, (half, window - 1 - half), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")
