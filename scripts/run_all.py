# scripts/run_all.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_module(module: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)

    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


def must_exist(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Also run a few run_one exemplar runs after the sweeps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for exemplar run_one calls (used if --example_seeds not provided).",
    )
    parser.add_argument(
        "--example_seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more seeds for exemplar run_one calls, e.g. --example_seeds 1 2 3",
    )
    args = parser.parse_args()

    # sweep configs
    cfg_sweep_dyad = "configs/sweep_dyad.json"
    cfg_sweep_triad = "configs/sweep_triad.json"
    cfg_sweep_classroom = "configs/sweep_classroom.json"

    # run_one configs (examples)
    # dyad examples: 3 distinct regimes
    cfg_dyad_antiphase = "configs/dyad_base_antiphase.json"
    cfg_dyad_inphase = "configs/dyad_base_inphase.json"
    cfg_dyad_drift = "configs/dyad_base_drift.json"

    # triad + classroom examples
    cfg_triad_all = "configs/triad_all_to_all.json"
    cfg_triad_coal = "configs/triad_coalition.json"
    cfg_triad_leader = "configs/triad_leader.json"
    cfg_classroom_base = "configs/classroom_base.json"

    for p in [
        cfg_sweep_dyad,
        cfg_sweep_triad,
        cfg_sweep_classroom,
        cfg_dyad_antiphase,
        cfg_dyad_inphase,
        cfg_dyad_drift,
        cfg_triad_all,
        cfg_triad_coal,
        cfg_triad_leader,
        cfg_classroom_base,
    ]:
        must_exist(p)

    # 1) sweeps
    run_module("scripts.run_sweep_dyad", ["--config", cfg_sweep_dyad])
    run_module("scripts.run_sweep_triad", ["--config", cfg_sweep_triad])
    run_module("scripts.run_sweep_classroom", ["--config", cfg_sweep_classroom])

    # 2) sweep figures (each script uses "latest sweep folder" logic)
    run_module("scripts.make_figures_dyad")
    run_module("scripts.make_figures_triad_sweep")
    run_module("scripts.make_figures_classroom_sweep")

    # 3) optional exemplar runs (good for the thesis narrative)
    if args.examples:
        example_seeds = args.example_seeds if args.example_seeds is not None else [args.seed]

        for s in example_seeds:
            print("\n" + "-" * 80)
            print(f"EXAMPLES: seed={s}")
            print("-" * 80)

            # dyad: three qualitatively different regimes
            run_module(
                "scripts.run_one",
                ["--scenario", "dyad", "--config", cfg_dyad_antiphase, "--seed", str(s)],
            )
            run_module(
                "scripts.run_one",
                ["--scenario", "dyad", "--config", cfg_dyad_inphase, "--seed", str(s)],
            )
            run_module(
                "scripts.run_one",
                ["--scenario", "dyad", "--config", cfg_dyad_drift, "--seed", str(s)],
            )

            # triad exemplars
            run_module(
                "scripts.run_one",
                ["--scenario", "triad", "--config", cfg_triad_all, "--seed", str(s)],
            )
            run_module(
                "scripts.run_one",
                ["--scenario", "triad", "--config", cfg_triad_coal, "--seed", str(s)],
            )
            run_module(
                "scripts.run_one",
                ["--scenario", "triad", "--config", cfg_triad_leader, "--seed", str(s)],
            )

            # classroom exemplar
            run_module(
                "scripts.run_one",
                ["--scenario", "classroom", "--config", cfg_classroom_base, "--seed", str(s)],
            )

    print("\nALL DONE ✅")


if __name__ == "__main__":
    main()
