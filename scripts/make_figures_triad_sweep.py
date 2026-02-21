# scripts/make_figures_triad_sweep.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_triad_sweep(outputs_dir: Path) -> Path:
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and "triad_sweep" in p.name]
    if not candidates:
        raise FileNotFoundError("No triad_sweep folders found in outputs/")
    return sorted(candidates)[-1]


def heatmap(df: pd.DataFrame, value: str, title: str, out_path: Path) -> None:
    # expects df filtered to one preset and one k_strong
    x = np.sort(df["delta_omega_tri"].unique())
    y = np.sort(df["k_weak"].unique())

    pivot = df.pivot(index="k_weak", columns="delta_omega_tri", values=value).reindex(index=y, columns=x)

    plt.figure(figsize=(7, 4.8))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(x)), [str(v) for v in x], rotation=45)
    plt.yticks(np.arange(len(y)), [str(v) for v in y])
    plt.xlabel("Δω (fixed mismatch)")
    plt.ylabel("k_weak")
    plt.title(title)
    plt.colorbar(label=value)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str, default=None)
    args = parser.parse_args()

    outputs = Path("outputs")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_triad_sweep(outputs)

    df = pd.read_csv(sweep_dir / "triad_sweep_conditions.csv")

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # make one heatmap per preset (for coalition_mean)
    for preset in sorted(df["preset"].unique()):
        sub = df[df["preset"] == preset].copy()

        # If you used multiple k_strong values, you can filter or loop them too
        # Here we just pick the first:
        k_strong_vals = sorted(sub["k_strong"].unique())
        sub = sub[sub["k_strong"] == k_strong_vals[0]]

        heatmap(
            sub,
            value="coalition_mean",
            title=f"Triad sweep ({preset}): coalition_mean",
            out_path=fig_dir / f"triad_{preset}_coalition_mean.png",
        )
        heatmap(
            sub,
            value="plv_pairmean_mean",
            title=f"Triad sweep ({preset}): mean pairwise PLV",
            out_path=fig_dir / f"triad_{preset}_plv_pairmean_mean.png",
        )
        heatmap(
            sub,
            value="r_mean",
            title=f"Triad sweep ({preset}): r_mean",
            out_path=fig_dir / f"triad_{preset}_r_mean.png",
        )
        heatmap(
            sub,
            value="leader_dom_mean",
            title=f"Triad sweep ({preset}): leader_dom_mean",
            out_path=fig_dir / f"triad_{preset}_leader_dom_mean.png",
        )
        heatmap(
            sub,
            value="leader_antiphase_dom_mean",
            title=f"Triad sweep ({preset}): leader antiphase dominance (distance-to-π)",
            out_path=fig_dir / f"triad_{preset}_leader_antiphase_dom_mean.png",
        )



    print(f"made triad sweep diagnostics in: {fig_dir}")


if __name__ == "__main__":
    main()
