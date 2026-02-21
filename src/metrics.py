# src/metrics.py
from __future__ import annotations
import numpy as np


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """
    Global Kuramoto order parameter R(t) in [0,1].
    theta shape: (t_steps, n)
    """
    complex_exponentials = np.exp(1j * theta)
    mean_vec = np.mean(complex_exponentials, axis=1)
    return np.abs(mean_vec)


def _rolling_mean_complex(z: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean over time of complex array z[t].
    Returns array length t_steps with NaN for t < window-1.
    """
    t_steps = z.size
    out = np.full(t_steps, np.nan, dtype=complex)
    if window <= 1:
        out[:] = z
        return out

    csum = np.cumsum(z, dtype=complex)
    out[window - 1] = csum[window - 1] / window
    for t in range(window, t_steps):
        out[t] = (csum[t] - csum[t - window]) / window
    return out


def plv_over_time(theta: np.ndarray, i: int, j: int, window: int) -> np.ndarray:
    """
    Windowed phase-locking value for pair (i,j).
    PLV(t) = |mean_{window} exp(i*(theta_i - theta_j))|
    """
    diff = theta[:, i] - theta[:, j]
    z = np.exp(1j * diff)
    z_bar = _rolling_mean_complex(z, window)
    return np.abs(z_bar)


def mean_phase_diff_over_time(theta: np.ndarray, i: int, j: int, window: int) -> np.ndarray:
    """
    Windowed circular mean of phase difference angle (in [-pi, pi]).
    Useful to distinguish in-phase (~0) vs anti-phase (~pi).
    """
    diff = theta[:, i] - theta[:, j]
    z = np.exp(1j * diff)
    z_bar = _rolling_mean_complex(z, window)
    return np.angle(z_bar)


def anti_phase_score_over_time(theta: np.ndarray, i: int, j: int, window: int) -> np.ndarray:
    """
    Score in [0,1] where 1 means mean phase difference is near pi (anti-phase).
    Uses circular distance to pi.
    """
    mu = mean_phase_diff_over_time(theta, i, j, window)

    # circular distance between mu and pi
    dist = np.angle(np.exp(1j * (mu - np.pi)))  # in [-pi, pi]
    return 1.0 - (np.abs(dist) / np.pi)


def mean_last_fraction(x: np.ndarray, frac: float = 0.3) -> float:
    """
    Mean over the last fraction of valid (non-NaN) samples.
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0,1]")
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return float("nan")
    n_last = max(1, int(np.floor(valid.size * frac)))
    return float(np.mean(valid[-n_last:]))

# src/metrics.py (additions)

def pairwise_plv_matrix(theta: np.ndarray, window: int) -> np.ndarray:
    """
    Pairwise windowed PLV matrix for all pairs.
    Returns (n, n) matrix using the LAST valid windowed PLV value for each pair.

    For i==j: returns 1.0 by definition.
    """
    n = theta.shape[1]
    out = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = 1.0
            else:
                plv_t = plv_over_time(theta, i=i, j=j, window=window)
                valid = plv_t[~np.isnan(plv_t)]
                out[i, j] = float(valid[-1]) if valid.size else float("nan")

    return out


def triad_coalition_index(plv_mat: np.ndarray) -> float:
    """
    Simple coalition index for triads:
      (max off-diagonal PLV) - (min off-diagonal PLV)

    Interpretation:
      - large value => one pair is much more locked than the weakest pair (coalition-like)
      - small value => more uniform coordination (either all locked similarly or all weak)
    """
    n = plv_mat.shape[0]
    off = []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(plv_mat[i, j]):
                off.append(plv_mat[i, j])
    if not off:
        return float("nan")
    return float(np.max(off) - np.min(off))

def pairwise_locked_angle(theta: np.ndarray, i: int, j: int, window: int) -> float:
    """
    Windowed circular mean phase difference angle (theta_i - theta_j) for the last valid time point.
    Returns angle in [-pi, pi]. This tells *where* the pair is locked (0=in-phase, pi=anti-phase, etc.).
    """
    mu_t = mean_phase_diff_over_time(theta, i=i, j=j, window=window)
    valid = mu_t[~np.isnan(mu_t)]
    return float(valid[-1]) if valid.size else float("nan")
