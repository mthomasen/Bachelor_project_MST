# src/sweep_triad.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.simulate import simulate_kuramoto
from src import scenarios
from src.metrics import (
    order_parameter,
    plv_over_time,
    mean_last_fraction,
    pairwise_plv_matrix,
    triad_coalition_index,
    pairwise_locked_angle,
)


@dataclass
class TriadSweepSpec:
    presets: list[str]
    delta_omega_values: list[float]
    k_strong_values: list[float]
    k_weak_values: list[float]
    seeds: list[int]

    k_leader_out: float = -1.0
    k_leader_in: float = -0.2
    k_all_value: float = -0.8

    dt: float = 0.01
    t_max: float = 120.0
    method: str = "rk4"
    normalize_by_n: bool = True
    noise_sd: float = 0.0

    mean_omega: float = 1.5
    sd_omega: float = 0.15
    omega_mode: str = "fixed_mismatch"

    plv_window_seconds: float = 2.0
    summary_last_frac: float = 0.3
    self_coupling: float = 0.0


def run_triad_sweep(spec: TriadSweepSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep triad presets and parameters, return:
      - runs_df: one row per run (preset, delta_omega, k_strong, k_weak, seed)
      - cond_df: aggregated per condition (means across seeds)
    """
    rows: list[dict] = []

    window = int(round(spec.plv_window_seconds / spec.dt))
    if window < 2:
        raise ValueError("plv_window_seconds too small for dt; increase window or decrease dt.")

    for preset in spec.presets:
        for delta_omega_tri in spec.delta_omega_values:
            for k_strong in spec.k_strong_values:
                for k_weak in spec.k_weak_values:
                    for seed in spec.seeds:
                        rng = np.random.default_rng(seed)

                        omega, theta0, k_matrix = scenarios.build_triad_preset(
                            rng=rng,
                            preset=preset,
                            mean_omega=spec.mean_omega,
                            sd_omega=spec.sd_omega,
                            omega_mode=spec.omega_mode,
                            delta_omega_tri=delta_omega_tri,

                            k_all=spec.k_all_value,
                            k_strong=k_strong,
                            k_weak=k_weak,
                            k_leader_out=spec.k_leader_out,
                            k_leader_in=spec.k_leader_in,

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
                        r_last = mean_last_fraction(r_t, frac=spec.summary_last_frac)

                        # pairwise summary
                        plv_mat_last = pairwise_plv_matrix(sim.theta, window=window)
                        plv_01 = float(plv_mat_last[0, 1])
                        plv_02 = float(plv_mat_last[0, 2])
                        plv_12 = float(plv_mat_last[1, 2])
                        plv_pairmean = float(np.mean([plv_01, plv_02, plv_12]))

                        coal = triad_coalition_index(plv_mat_last)
                        leader_dom = float(((plv_01 + plv_02) / 2.0) - plv_12)

                        # locked angles (last window)
                        mu_01 = pairwise_locked_angle(sim.theta, 0, 1, window=window)
                        mu_02 = pairwise_locked_angle(sim.theta, 0, 2, window=window)
                        mu_12 = pairwise_locked_angle(sim.theta, 1, 2, window=window)
                        def _circ_dist(mu: float, target: float) -> float:
                            if np.isnan(mu):
                                return float("nan")
                            return float(np.abs(np.angle(np.exp(1j * (mu - target)))))  # [0, pi]

                        d01 = _circ_dist(mu_01, np.pi)
                        d02 = _circ_dist(mu_02, np.pi)
                        d12 = _circ_dist(mu_12, np.pi)

                        leader_antiphase_dom = float(d12 - 0.5 * (d01 + d02))

                        rows.append(
                            {
                                "preset": preset,
                                "delta_omega_tri": delta_omega_tri,
                                "k_strong": k_strong,
                                "k_weak": k_weak,
                                "seed": seed,
                                "dt": spec.dt,
                                "t_max": spec.t_max,
                                "plv_window_seconds": spec.plv_window_seconds,
                                "r_mean_last": r_last,
                                "plv_01_last": plv_01,
                                "plv_02_last": plv_02,
                                "plv_12_last": plv_12,
                                "plv_pairmean_last": plv_pairmean,
                                "coalition_index": coal,
                                "leader_dominance": leader_dom,
                                "leader_antiphase_dom_last": leader_antiphase_dom,
                                "d01_last": d01,
                                "d02_last": d02,
                                "d12_last": d12,
                                "mu_01_last": mu_01,
                                "mu_02_last": mu_02,
                                "mu_12_last": mu_12,
                            }
                        )

    runs_df = pd.DataFrame(rows)

    cond_df = (
        runs_df
        .groupby(["preset", "delta_omega_tri", "k_strong", "k_weak"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            r_mean=("r_mean_last", "mean"),
            r_sd=("r_mean_last", "std"),
            plv_pairmean_mean=("plv_pairmean_last", "mean"),
            plv_pairmean_sd=("plv_pairmean_last", "std"),
            coalition_mean=("coalition_index", "mean"),
            coalition_sd=("coalition_index", "std"),
            leader_dom_mean=("leader_dominance", "mean"),
            leader_dom_sd=("leader_dominance", "std"),
            leader_antiphase_dom_mean=("leader_antiphase_dom_last", "mean"),
            leader_antiphase_dom_sd=("leader_antiphase_dom_last", "std")
        )
    )

    return runs_df, cond_df
