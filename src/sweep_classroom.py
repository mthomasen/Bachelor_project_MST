# src/sweep_classroom.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.simulate import simulate_kuramoto
from src import scenarios
from src.metrics import order_parameter, plv_over_time, mean_last_fraction, pairwise_locked_angle


@dataclass
class ClassroomSweepSpec:
    n_students: int
    seeds: list[int]

    # sweep grids
    k_ts_values: list[float]  # teacher -> student
    k_st_values: list[float]  # student -> teacher
    k_ss_values: list[float]  # student <-> student

    # simulation params
    dt: float
    t_max: float
    method: str
    normalize_by_n: bool
    noise_sd: float

    # omega sampling
    mean_omega: float
    sd_omega: float
    self_coupling: float = 0.0

    # metrics
    plv_window_seconds: float = 2.0
    summary_last_frac: float = 0.3


def _teacher_student_plv_last(theta: np.ndarray, window: int, teacher: int = 0) -> tuple[float, list[float]]:
    """
    Return (mean PLV teacher-student, list of student PLVs) using last valid PLV value.
    theta shape: (t_steps, n)
    """
    n = theta.shape[1]
    students = [i for i in range(n) if i != teacher]
    plvs = []
    for s in students:
        plv_t = plv_over_time(theta, teacher, s, window=window)
        valid = plv_t[~np.isnan(plv_t)]
        plvs.append(float(valid[-1]) if valid.size else float("nan"))
    return float(np.nanmean(plvs)), plvs


def _student_student_plv_last(theta: np.ndarray, window: int, teacher: int = 0) -> float:
    """
    Mean PLV across all student-student pairs (last valid windowed PLV).
    """
    n = theta.shape[1]
    students = [i for i in range(n) if i != teacher]
    vals = []
    for i_idx in range(len(students)):
        for j_idx in range(i_idx + 1, len(students)):
            i = students[i_idx]
            j = students[j_idx]
            plv_t = plv_over_time(theta, i, j, window=window)
            valid = plv_t[~np.isnan(plv_t)]
            vals.append(float(valid[-1]) if valid.size else float("nan"))
    return float(np.nanmean(vals))

def _circ_dist(mu: float, target: float) -> float:
    """
    Circular distance |wrap(mu - target)| in [0, pi].
    Returns NaN if mu is NaN.
    """
    if np.isnan(mu):
        return float("nan")
    return float(np.abs(np.angle(np.exp(1j * (mu - target)))))

def _teacher_student_distpi_mean(theta: np.ndarray, window: int, teacher: int = 0) -> float:
    """
    Mean circular distance to anti-phase (pi) for teacher-student locked angles.
    Smaller = closer to anti-phase.
    """
    n = theta.shape[1]
    students = [i for i in range(n) if i != teacher]
    ds = []
    for s in students:
        mu = pairwise_locked_angle(theta, teacher, s, window=window)
        ds.append(_circ_dist(mu, np.pi))
    return float(np.nanmean(ds))


def _student_student_distpi_mean(theta: np.ndarray, window: int, teacher: int = 0) -> float:
    """
    Mean circular distance to anti-phase (pi) for student-student locked angles.
    Smaller = closer to anti-phase.
    """
    n = theta.shape[1]
    students = [i for i in range(n) if i != teacher]
    ds = []
    for a in range(len(students)):
        for b in range(a + 1, len(students)):
            i = students[a]
            j = students[b]
            mu = pairwise_locked_angle(theta, i, j, window=window)
            ds.append(_circ_dist(mu, np.pi))
    return float(np.nanmean(ds))


def run_classroom_sweep(spec: ClassroomSweepSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    window = int(round(spec.plv_window_seconds / spec.dt))
    if window < 2:
        raise ValueError("plv_window_seconds too small for dt; increase window or decrease dt.")

    for k_ts in spec.k_ts_values:
        for k_st in spec.k_st_values:
            for k_ss in spec.k_ss_values:
                for seed in spec.seeds:
                    rng = np.random.default_rng(seed)

                    omega, theta0, k_matrix = scenarios.build_classroom(
                        rng=rng,
                        n_students=spec.n_students,
                        mean_omega=spec.mean_omega,
                        sd_omega=spec.sd_omega,
                        k_ts=k_ts,
                        k_st=k_st,
                        k_ss=k_ss,
                        self_coupling=spec.self_coupling,
                    )

                    sim = simulate_kuramoto(
                        omega=omega,
                        theta0=theta0,
                        k_matrix=k_matrix,
                        dt=spec.dt,
                        t_max=spec.t_max,
                        seed=seed,
                        noise_sd=spec.noise_sd,
                        normalize_by_n=spec.normalize_by_n,
                        method=spec.method,
                    )

                    r_t = order_parameter(sim.theta)
                    r_mean_last = mean_last_fraction(r_t, frac=spec.summary_last_frac)

                    ts_mean_plv_last, ts_plvs = _teacher_student_plv_last(sim.theta, window=window, teacher=0)
                    ss_mean_plv_last = _student_student_plv_last(sim.theta, window=window, teacher=0)
                    ts_distpi_mean_last = _teacher_student_distpi_mean(sim.theta, window=window, teacher=0)
                    ss_distpi_mean_last = _student_student_distpi_mean(sim.theta, window=window, teacher=0)

                    # positive => teacher-student is CLOSER to anti-phase than student-student
                    teacher_antiphase_dominance_last = float(ss_distpi_mean_last - ts_distpi_mean_last)

                    # teacher dominance: teacher-student coupling coherence vs student-student coherence
                    teacher_dominance = float(ts_mean_plv_last - ss_mean_plv_last)

                    rows.append(
                        {
                            "seed": seed,
                            "n_students": spec.n_students,
                            "dt": spec.dt,
                            "t_max": spec.t_max,
                            "plv_window_seconds": spec.plv_window_seconds,
                            "k_ts": k_ts,
                            "k_st": k_st,
                            "k_ss": k_ss,
                            "r_mean_last": r_mean_last,
                            "ts_plv_mean_last": ts_mean_plv_last,
                            "ss_plv_mean_last": ss_mean_plv_last,
                            "teacher_dominance_last": teacher_dominance,
                            "ts_distpi_mean_last": ts_distpi_mean_last,
                            "ss_distpi_mean_last": ss_distpi_mean_last,
                            "teacher_antiphase_dom_last": teacher_antiphase_dominance_last,
                        }
                    )

    runs_df = pd.DataFrame(rows)

    cond_df = (
        runs_df
        .groupby(["k_ts", "k_st", "k_ss"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            r_mean=("r_mean_last", "mean"),
            r_sd=("r_mean_last", "std"),
            ts_plv_mean=("ts_plv_mean_last", "mean"),
            ts_plv_sd=("ts_plv_mean_last", "std"),
            ss_plv_mean=("ss_plv_mean_last", "mean"),
            ss_plv_sd=("ss_plv_mean_last", "std"),
            teacher_dom_mean=("teacher_dominance_last", "mean"),
            teacher_dom_sd=("teacher_dominance_last", "std"),
            ts_distpi_mean=("ts_distpi_mean_last", "mean"),
            ts_distpi_sd=("ts_distpi_mean_last", "std"),
            ss_distpi_mean=("ss_distpi_mean_last", "mean"),
            ss_distpi_sd=("ss_distpi_mean_last", "std"),
            teacher_antiphase_dom_mean=("teacher_antiphase_dom_last", "mean"),
            teacher_antiphase_dom_sd=("teacher_antiphase_dom_last", "std"),
        )
    )

    return runs_df, cond_df
