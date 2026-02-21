# src/sweep_dyad.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.simulate import simulate_kuramoto
from src import scenarios
from src.metrics import order_parameter, plv_over_time, anti_phase_score_over_time, mean_last_fraction


@dataclass
class DyadSweepSpec:
    k_values: list[float]
    delta_omega_values: list[float]
    seeds: list[int]
    dt: float
    t_max: float
    method: str
    normalize_by_n: bool
    noise_sd: float
    plv_window_seconds: float
    summary_last_frac: float
    mean_omega: float
    sd_omega: float
    self_coupling: float = 0.0


def run_dyad_sweep(spec: DyadSweepSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs dyad simulations across (k, delta_omega) and seeds.

    Returns
    -------
    runs_df: one row per run (k, delta_omega, seed)
    cond_df: aggregated per (k, delta_omega) across seeds (mean + sd + n)
    """
    rows: list[dict] = []
    pair = (0, 1)

    window = int(round(spec.plv_window_seconds / spec.dt))
    if window < 2:
        raise ValueError("plv_window_seconds is too small relative to dt; window must be >= 2 steps")

    for k in spec.k_values:
        for delta_omega in spec.delta_omega_values:
            for seed in spec.seeds:
                rng = np.random.default_rng(seed)

                omega, theta0, k_matrix = scenarios.build_dyad(
                    rng=rng,
                    mean_omega=spec.mean_omega,
                    sd_omega=spec.sd_omega,
                    delta_omega=delta_omega,
                    k=k,
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
                plv_t = plv_over_time(sim.theta, pair[0], pair[1], window=window)
                anti_t = anti_phase_score_over_time(sim.theta, pair[0], pair[1], window=window)

                row = {
                    "k": k,
                    "delta_omega": delta_omega,
                    "seed": seed,
                    "dt": spec.dt,
                    "t_max": spec.t_max,
                    "plv_window_seconds": spec.plv_window_seconds,
                    "r_mean_last": mean_last_fraction(r_t, frac=spec.summary_last_frac),
                    "plv_mean_last": mean_last_fraction(plv_t, frac=spec.summary_last_frac),
                    "anti_mean_last": mean_last_fraction(anti_t, frac=spec.summary_last_frac),
                    "omega_0": float(omega[0]),
                    "omega_1": float(omega[1]),
                }
                rows.append(row)

    runs_df = pd.DataFrame(rows)

    cond_df = (
        runs_df
        .groupby(["k", "delta_omega"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            r_mean=("r_mean_last", "mean"),
            r_sd=("r_mean_last", "std"),
            plv_mean=("plv_mean_last", "mean"),
            plv_sd=("plv_mean_last", "std"),
            anti_mean=("anti_mean_last", "mean"),
            anti_sd=("anti_mean_last", "std"),
        )
    )

    return runs_df, cond_df
