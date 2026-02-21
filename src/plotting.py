# src/plotting.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_phases(time: np.ndarray, theta: np.ndarray, title: str, path: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    n = theta.shape[1]
    for i in range(n):
        plt.plot(time, theta[:, i], label=f"theta_{i}")
    plt.xlabel("time")
    plt.ylabel("phase (rad)")
    plt.title(title)
    if n <= 10:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_time_series(time: np.ndarray, y: np.ndarray, title: str, ylab: str, path: str | None = None) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(time, y)
    plt.xlabel("time")
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_phase_diff_polar(theta: np.ndarray, i: int, j: int, title: str, bins: int = 24, path: str | None = None) -> None:
    """
    Polar histogram ("rose plot") of phase differences theta_i - theta_j over time.
    """
    diff = (theta[:, i] - theta[:, j] + np.pi) % (2 * np.pi) - np.pi

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.hist(diff, bins=bins)
    ax.set_title(title)
    if path is not None:
        plt.savefig(path, dpi=200)
        plt.close()
    else:
        plt.show()
