# scripts/run_sweep_classroom.py
from __future__ import annotations
import argparse

from src.io import make_run_dir, load_config, save_config
from src.sweep_classroom import ClassroomSweepSpec, run_classroom_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    spec = ClassroomSweepSpec(
        n_students=int(config["n_students"]),
        seeds=list(config["seeds"]),
        k_ts_values=list(config["k_ts_values"]),
        k_st_values=list(config["k_st_values"]),
        k_ss_values=list(config["k_ss_values"]),
        dt=float(config["dt"]),
        t_max=float(config["t_max"]),
        method=str(config.get("method", "rk4")),
        normalize_by_n=bool(config.get("normalize_by_n", True)),
        noise_sd=float(config.get("noise_sd", 0.0)),
        mean_omega=float(config["mean_omega"]),
        sd_omega=float(config["sd_omega"]),
        self_coupling=float(config.get("self_coupling", 0.0)),
        plv_window_seconds=float(config["plv_window_seconds"]),
        summary_last_frac=float(config.get("summary_last_frac", 0.3)),
    )

    run_dir = make_run_dir(tag="classroom_sweep")
    save_config(config, run_dir / "config_used.json")

    runs_df, cond_df = run_classroom_sweep(spec)
    runs_df.to_csv(run_dir / "classroom_sweep_runs.csv", index=False)
    cond_df.to_csv(run_dir / "classroom_sweep_conditions.csv", index=False)

    print(f"saved to: {run_dir}")


if __name__ == "__main__":
    main()
