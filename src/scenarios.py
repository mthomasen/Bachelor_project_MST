# src/scenarios.py
from __future__ import annotations
import numpy as np


def sample_omega_normal(rng: np.random.Generator, n: int, mean: float, sd: float) -> np.ndarray:
    return rng.normal(mean, sd, size=n)


def sample_theta_uniform(rng: np.random.Generator, n: int, low: float = -np.pi / 4, high: float = np.pi / 4) -> np.ndarray:
    return rng.uniform(low, high, size=n)


def build_dyad(
    rng: np.random.Generator,
    mean_omega: float = 1.5,
    sd_omega: float = 0.15,
    delta_omega: float = 0.0,
    k: float = -1.0,
    self_coupling: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dyad with controllable frequency mismatch (delta_omega) and symmetric coupling k.

    omega = [mean+delta/2, mean-delta/2] + optional noise via sd_omega
    k_matrix has off-diagonal entries = k, diagonal = self_coupling
    """
    base = sample_omega_normal(rng, 2, mean_omega, sd_omega)
    # impose mismatch deterministically on top of sampled base mean
    omega = np.array([np.mean(base) + delta_omega / 2.0, np.mean(base) - delta_omega / 2.0], dtype=float)

    theta0 = sample_theta_uniform(rng, 2)

    k_matrix = np.array(
        [[self_coupling, k],
         [k, self_coupling]],
        dtype=float
    )
    return omega, theta0, k_matrix


def build_triad_all_to_all(
    rng: np.random.Generator,
    mean_omega: float = 1.5,
    sd_omega: float = 0.15,
    k_offdiag: float = -0.8,
    self_coupling: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triad with uniform all-to-all coupling (off-diagonal constant).
    """
    omega = sample_omega_normal(rng, 3, mean_omega, sd_omega)
    theta0 = sample_theta_uniform(rng, 3)

    k_matrix = np.full((3, 3), k_offdiag, dtype=float)
    np.fill_diagonal(k_matrix, self_coupling)
    return omega, theta0, k_matrix


def build_triad_custom(
    rng: np.random.Generator,
    omega: np.ndarray,
    k_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use explicit omega and k_matrix (for reporting specific topologies).
    """
    omega = np.asarray(omega, dtype=float)
    k_matrix = np.asarray(k_matrix, dtype=float)
    if omega.size != 3 or k_matrix.shape != (3, 3):
        raise ValueError("custom triad requires omega shape (3,) and k_matrix shape (3,3)")
    theta0 = sample_theta_uniform(rng, 3)
    return omega, theta0, k_matrix


def build_classroom(
    rng: np.random.Generator,
    n_students: int = 10,
    mean_omega: float = 1.5,
    sd_omega: float = 0.15,
    k_ts: float = -0.8,   # teacher -> student
    k_st: float = -0.2,   # student -> teacher
    k_ss: float = -0.3,   # student <-> student
    self_coupling: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classroom topology:
      index 0 is teacher
      indices 1..n_students are students

    Convention: k_matrix[i, j] is influence from j to i.

    Therefore:
      teacher -> student means: k_matrix[student, teacher] = k_ts
      student -> teacher means: k_matrix[teacher, student] = k_st
      student <-> student means: k_matrix[si, sj] = k_ss for si != sj
    """
    n = 1 + n_students
    omega = sample_omega_normal(rng, n, mean_omega, sd_omega)
    theta0 = sample_theta_uniform(rng, n)

    k_matrix = np.zeros((n, n), dtype=float)
    np.fill_diagonal(k_matrix, self_coupling)

    teacher = 0
    students = np.arange(1, n)

    # student <-> student
    for i in students:
        for j in students:
            if i != j:
                k_matrix[i, j] = k_ss

    # teacher -> student (teacher is column, student is row)
    for s in students:
        k_matrix[s, teacher] = k_ts

    # student -> teacher (student is column, teacher is row)
    for s in students:
        k_matrix[teacher, s] = k_st

    return omega, theta0, k_matrix


# src/scenarios.py  (additions)

def build_triad_preset(
    rng: np.random.Generator,
    preset: str,
    mean_omega: float = 1.5,
    sd_omega: float = 0.15,
    omega_mode: str = "random",
    delta_omega_tri: float = 0.0,
    k_all: float = -0.8,
    k_strong: float = -1.2,
    k_weak: float = -0.2,
    k_leader_out: float = -1.0,
    k_leader_in: float = -0.2,
    self_coupling: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triad presets for RQ2. Returns (omega, theta0, k_matrix).

    Convention: k_matrix[i, j] is influence from j to i (row receives from column).

    Presets
    -------
    - "all_to_all": uniform off-diagonal coupling = k_all
    - "coalition_01": oscillators 0 and 1 strongly coupled to each other; links to/from 2 are weak
    - "leader_0": oscillator 0 drives others (strong outgoing), but others weakly influence 0 (asymmetric)
    """
    if omega_mode == "random":
        omega = sample_omega_normal(rng, 3, mean_omega, sd_omega)
    elif omega_mode == "fixed_mismatch":
        omega = np.array([mean_omega + delta_omega_tri / 2.0,
                      mean_omega - delta_omega_tri / 2.0,
                      mean_omega], dtype=float)
    else:
        raise ValueError("omega_mode must be 'random' or 'fixed_mismatch'")
    theta0 = sample_theta_uniform(rng, 3)

    preset = preset.lower()

    if preset == "all_to_all":
        k_matrix = np.full((3, 3), k_all, dtype=float)
        np.fill_diagonal(k_matrix, self_coupling)

    elif preset == "coalition_01":
        # Start with weak all-to-all links
        k_matrix = np.full((3, 3), k_weak, dtype=float)
        np.fill_diagonal(k_matrix, self_coupling)

        # Make 0 <-> 1 strong (both directions)
        k_matrix[0, 1] = k_strong  # 1 -> 0
        k_matrix[1, 0] = k_strong  # 0 -> 1

        # Links involving oscillator 2 remain weak (k_weak)

    elif preset == "leader_0":
        k_matrix = np.full((3, 3), k_weak, dtype=float)
        np.fill_diagonal(k_matrix, self_coupling)

        # "0 drives others": strong influence from 0 to 1 and 2
        k_matrix[1, 0] = k_leader_out  # 0 -> 1
        k_matrix[2, 0] = k_leader_out  # 0 -> 2

        # weak feedback from 1 and 2 to 0
        k_matrix[0, 1] = k_leader_in   # 1 -> 0
        k_matrix[0, 2] = k_leader_in   # 2 -> 0

        # keep 1 <-> 2 weak
        k_matrix[1, 2] = k_weak
        k_matrix[2, 1] = k_weak

    else:
        raise ValueError(f"Unknown triad preset: {preset}")

    return omega, theta0, k_matrix
