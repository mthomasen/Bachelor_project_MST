# scripts/run_one.py
from __future__ import annotations
import argparse
import numpy as np

from src.simulate import simulate_kuramoto
from src import scenarios
from src.metrics import (
    order_parameter,
    plv_over_time,
    anti_phase_score_over_time,
    mean_last_fraction,
    pairwise_plv_matrix,
    triad_coalition_index,
    pairwise_locked_angle,
)
from src.plotting import plot_phases, plot_time_series, plot_phase_diff_polar
from src.io import (
    make_run_dir,
    load_config,
    save_config,
    save_theta_csv,
    save_metrics_timeseries,
    save_metrics_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scenario", type=str, choices=["dyad", "triad", "classroom"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    # build scenario
    if args.scenario == "dyad":
        omega, theta0, k_matrix = scenarios.build_dyad(
            rng=rng,
            mean_omega=config["mean_omega"],
            sd_omega=config["sd_omega"],
            delta_omega=config["delta_omega"],
            k=config["k"],
            self_coupling=config.get("self_coupling", 0.0),
        )
        pair = (0, 1)
        tag = f"dyad_k{config['k']}_dw{config['delta_omega']}_seed{args.seed}"

    elif args.scenario == "triad":
        omega, theta0, k_matrix = scenarios.build_triad_preset(
            rng=rng,
            preset=config["triad_preset"],
            mean_omega=config["mean_omega"],
            sd_omega=config["sd_omega"],
            omega_mode=config.get("omega_mode", "random"),
            delta_omega_tri=config.get("delta_omega_tri", 0.0),
            k_all=config.get("k_all", -0.8),
            k_strong=config.get("k_strong", -1.2),
            k_weak=config.get("k_weak", -0.2),
            k_leader_out=config.get("k_leader_out", -1.0),
            k_leader_in=config.get("k_leader_in", -0.2),
            self_coupling=config.get("self_coupling", 0.0),
        )
        pair = (0, 1)
        tag = (
            f"triad_{config['triad_preset']}"
            f"_om{config.get('omega_mode','random')}"
            f"_dw{config.get('delta_omega_tri',0.0)}"
            f"_seed{args.seed}"
        )

    else:  # classroom
        omega, theta0, k_matrix = scenarios.build_classroom(
            rng=rng,
            n_students=config["n_students"],
            mean_omega=config["mean_omega"],
            sd_omega=config["sd_omega"],
            k_ts=config["k_ts"],
            k_st=config["k_st"],
            k_ss=config["k_ss"],
            self_coupling=config.get("self_coupling", 0.0),
        )
        pair = (0, 1)  # teacher-student
        tag = f"classroom_ts{config['k_ts']}_st{config['k_st']}_ss{config['k_ss']}_seed{args.seed}"

    # simulate
    sim = simulate_kuramoto(
        omega=omega,
        theta0=theta0,
        k_matrix=k_matrix,
        dt=config["dt"],
        t_max=config["t_max"],
        seed=args.seed,
        noise_sd=config.get("noise_sd", 0.0),
        normalize_by_n=config.get("normalize_by_n", True),
        method=config.get("method", "rk4"),
    )

    # metrics
    window = int(round(config["plv_window_seconds"] / config["dt"]))
    if window < 2:
        raise ValueError("plv_window_seconds too small for dt; increase window or decrease dt.")

    r_t = order_parameter(sim.theta)
    plv_t = plv_over_time(sim.theta, pair[0], pair[1], window=window)
    anti_t = anti_phase_score_over_time(sim.theta, pair[0], pair[1], window=window)

    metrics_ts = {
        "r": r_t,
        "plv_pair": plv_t,
        "anti_phase_score_pair": anti_t,
    }

    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "n": sim.theta.shape[1],
        "dt": sim.dt,
        "t_max": float(config["t_max"]),
        "plv_window_seconds": float(config["plv_window_seconds"]),
        "r_mean_last": mean_last_fraction(r_t, frac=config.get("summary_last_frac", 0.3)),
        "plv_mean_last": mean_last_fraction(plv_t, frac=config.get("summary_last_frac", 0.3)),
        "anti_mean_last": mean_last_fraction(anti_t, frac=config.get("summary_last_frac", 0.3)),
    }

    # triad-specific summary extras (coalitions / leadership / antiphase distances)
    if args.scenario == "triad":
        plv_mat_last = pairwise_plv_matrix(sim.theta, window=window)

        plv_01 = float(plv_mat_last[0, 1])
        plv_02 = float(plv_mat_last[0, 2])
        plv_12 = float(plv_mat_last[1, 2])

        leader_dominance = float(((plv_01 + plv_02) / 2.0) - plv_12)

        mu_01 = pairwise_locked_angle(sim.theta, 0, 1, window=window)
        mu_02 = pairwise_locked_angle(sim.theta, 0, 2, window=window)
        mu_12 = pairwise_locked_angle(sim.theta, 1, 2, window=window)

        def _circ_dist(mu: float, target: float) -> float:
            if np.isnan(mu):
                return float("nan")
            return float(np.abs(np.angle(np.exp(1j * (mu - target)))))  # in [0, pi]

        d01 = _circ_dist(mu_01, np.pi)
        d02 = _circ_dist(mu_02, np.pi)
        d12 = _circ_dist(mu_12, np.pi)

        leader_antiphase_dom = float(d12 - 0.5 * (d01 + d02))

        summary.update(
            {
                "triad_preset": str(config["triad_preset"]),
                "plv_01_last": plv_01,
                "plv_02_last": plv_02,
                "plv_12_last": plv_12,
                "coalition_index": triad_coalition_index(plv_mat_last),
                "leader_dominance": leader_dominance,
                "mu_01_last": float(mu_01),
                "mu_02_last": float(mu_02),
                "mu_12_last": float(mu_12),
                "d01_last": float(d01),
                "d02_last": float(d02),
                "d12_last": float(d12),
                "leader_antiphase_dom_last": float(leader_antiphase_dom),
            }
        )

    # save
    run_dir = make_run_dir(tag=tag)
    save_config({**config, "scenario": args.scenario, "seed": args.seed}, run_dir / "config_used.json")
    save_theta_csv(sim.time, sim.theta, run_dir / "theta.csv")
    save_metrics_timeseries(sim.time, metrics_ts, run_dir / "metrics_timeseries.csv")
    save_metrics_summary(summary, run_dir / "metrics_summary.csv")

    # figures
    plot_phases(sim.time, sim.theta, title=f"phases: {tag}", path=str(run_dir / "figures" / "phases.png"))
    plot_time_series(
        sim.time, r_t, title=f"order parameter R(t): {tag}", ylab="R(t)", path=str(run_dir / "figures" / "r_t.png")
    )
    plot_time_series(
        sim.time,
        plv_t,
        title=f"PLV (windowed) for pair {pair[0]}-{pair[1]}: {tag}",
        ylab="plv",
        path=str(run_dir / "figures" / "plv_t.png"),
    )

    # polar plots
    if args.scenario == "triad":
        plot_phase_diff_polar(sim.theta, 0, 1, title=f"phase diff polar 0-1: {tag}", path=str(run_dir / "figures" / "phase_diff_01.png"))
        plot_phase_diff_polar(sim.theta, 0, 2, title=f"phase diff polar 0-2: {tag}", path=str(run_dir / "figures" / "phase_diff_02.png"))
        plot_phase_diff_polar(sim.theta, 1, 2, title=f"phase diff polar 1-2: {tag}", path=str(run_dir / "figures" / "phase_diff_12.png"))
    else:
        plot_phase_diff_polar(sim.theta, pair[0], pair[1], title=f"phase diff polar: {tag}", path=str(run_dir / "figures" / "phase_diff_polar.png"))

    print(f"saved to: {run_dir}")


if __name__ == "__main__":
    main()
