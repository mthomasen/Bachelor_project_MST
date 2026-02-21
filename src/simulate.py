#src/simulate.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    time: np.ndarray        # shape (t_steps,)
    theta: np.ndarray       # shape (t_steps, n)
    omega: np.ndarray       # shape (n,)
    k_matrix: np.ndarray    # shape (n, n)
    dt: float
    seed: int | None

def kuramoto_rhs(theta: np.ndarray, omega: np.ndarray, k_matrix: np.ndarray, normalize_by_n: bool) -> np.ndarray:
    """
    Right-hand side of a general (possibly asymmetric) Kuramoto-type model.

    Model convention used throughout this project:
        dtheta_i/dt = omega_i + sum_j k_ij * sin(theta_j - theta_i)

    Notes:
    - k_ij is influence FROM j TO i (row i receives from column j).
    - If normalize_by_n is True, the coupling sum is divided by n.
    """
    n = theta.size
    phase_diff = theta[None, :] - theta[:, None]  # (n, n): theta_j - theta_i
    coupling_term = np.sum(k_matrix * np.sin(phase_diff), axis=1)  # sum over j
    if normalize_by_n:
        coupling_term = coupling_term / n
    return omega + coupling_term


def simulate_kuramoto(
    omega: np.ndarray,
    theta0: np.ndarray,
    k_matrix: np.ndarray,
    dt: float,
    t_max: float,
    seed: int | None = None,
    noise_sd: float = 0.0,
    normalize_by_n: bool = True,
    method: str = "rk4",
) -> SimulationResult:
    """
    Simulate coupled phase oscillators using a fixed-step integrator.

    Parameters
    ----------
    omega : (n,)
        Natural frequencies.
    theta0 : (n,)
        Initial phases.
    k_matrix : (n, n)
        Coupling matrix. k_matrix[i, j] is influence from j to i.
    dt : float
        Time step.
    t_max : float
        Total simulated time.
    seed : int | None
        RNG seed for reproducibility (affects noise only).
    noise_sd : float
        Standard deviation of additive noise on dtheta (per sqrt(time)).
        Implemented as: theta_{t+dt} += noise_sd * sqrt(dt) * N(0,1)
    normalize_by_n : bool
        If True, divide coupling term by n (common Kuramoto scaling).
    method : {"rk4", "euler"}
        Integration method.

    Returns
    -------
    SimulationResult with time grid and theta trajectories.
    """
    omega = np.asarray(omega, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)
    k_matrix = np.asarray(k_matrix, dtype=float)

    n = omega.size
    if theta0.size != n:
        raise ValueError("theta0 must have same length as omega")
    if k_matrix.shape != (n, n):
        raise ValueError("k_matrix must be shape (n, n)")

    t_steps = int(np.floor(t_max / dt)) + 1
    time = np.linspace(0.0, dt * (t_steps - 1), t_steps)

    theta = np.zeros((t_steps, n), dtype=float)
    theta[0] = theta0

    rng = np.random.default_rng(seed)

    def rhs(x: np.ndarray) -> np.ndarray:
        return kuramoto_rhs(x, omega, k_matrix, normalize_by_n)

    for t in range(t_steps - 1):
        x = theta[t]

        if method == "euler":
            dx = rhs(x)
            x_next = x + dt * dx
        elif method == "rk4":
            k1 = rhs(x)
            k2 = rhs(x + 0.5 * dt * k1)
            k3 = rhs(x + 0.5 * dt * k2)
            k4 = rhs(x + dt * k3)
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError("method must be 'rk4' or 'euler'")

        if noise_sd > 0.0:
            x_next = x_next + noise_sd * np.sqrt(dt) * rng.normal(0.0, 1.0, size=n)

        theta[t + 1] = x_next

    return SimulationResult(
        time=time,
        theta=theta,
        omega=omega,
        k_matrix=k_matrix,
        dt=dt,
        seed=seed,
    )