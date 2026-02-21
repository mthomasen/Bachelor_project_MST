# scripts/run_sweep_triad.py
from __future__ import annotations
import argparse
from pathlib import Path

from src.io import make_run_dir, load_config, save_config
from src.sweep_triad import TriadSweepSpec, run_triad_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to triad sweep config JSON")
    args = parser.parse_args()

    config = load_config(args.config)

    spec = TriadSweepSpec(
        presets=list(config["presets"]),
        delta_omega_values=list(config["delta_omega_tri_values"]),
        k_strong_values=list(config["k_strong_values"]),
        k_weak_values=list(config["k_weak_values"]),
        k_leader_out=float(config.get("k_leader_out", -1.0)),
        k_leader_in=float(config.get("k_leader_in", -0.2)),
        k_all_value=float(config.get("k_all_value", -0.8)),
        seeds=list(config["seeds"]),
        dt=float(config["dt"]),
        t_max=float(config["t_max"]),
        method=str(config.get("method", "rk4")),
        normalize_by_n=bool(config.get("normalize_by_n", True)),
        noise_sd=float(config.get("noise_sd", 0.0)),
        mean_omega=float(config["mean_omega"]),
        sd_omega=float(config.get("sd_omega", 0.15)),
        omega_mode=str(config.get("omega_mode", "fixed_mismatch")),
        plv_window_seconds=float(config["plv_window_seconds"]),
        summary_last_frac=float(config.get("summary_last_frac", 0.3)),
        self_coupling=float(config.get("self_coupling", 0.0)),
    )

    run_dir = make_run_dir(tag="triad_sweep")
    save_config(config, run_dir / "config_used.json")

    runs_df, cond_df = run_triad_sweep(spec)
    runs_df.to_csv(run_dir / "triad_sweep_runs.csv", index=False)
    cond_df.to_csv(run_dir / "triad_sweep_conditions.csv", index=False)

    print(f"saved to: {run_dir}")


if __name__ == "__main__":
    main()
