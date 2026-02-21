# scripts/make_figures.py
from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_sweep_dir(outputs_dir: Path) -> Path:
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and "dyad_sweep" in p.name]
    if not candidates:
        raise FileNotFoundError("No dyad_sweep folders found in outputs/")
    return sorted(candidates)[-1]


def heatmap_from_conditions(
    cond_df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
) -> None:
    ks = np.sort(cond_df["k"].unique())
    dws = np.sort(cond_df["delta_omega"].unique())

    pivot = cond_df.pivot(index="delta_omega", columns="k", values=value_col).reindex(index=dws, columns=ks)

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(ticks=np.arange(len(ks)), labels=[str(k) for k in ks], rotation=45)
    plt.yticks(ticks=np.arange(len(dws)), labels=[str(dw) for dw in dws])
    plt.xlabel("coupling k")
    plt.ylabel("frequency mismatch Δω")
    plt.title(title)
    plt.colorbar(label=value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str, default=None, help="Path to an outputs/*dyad_sweep folder")
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_sweep_dir(outputs_dir)

    cond_path = sweep_dir / "dyad_sweep_conditions.csv"
    if not cond_path.exists():
        raise FileNotFoundError(f"Missing {cond_path}")

    cond_df = pd.read_csv(cond_path)

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    heatmap_from_conditions(
        cond_df,
        value_col="anti_mean",
        title="Dyad regime map: anti-phase score (mean over last window)",
        out_path=fig_dir / "dyad_heatmap_antiphase.png",
    )

    heatmap_from_conditions(
        cond_df,
        value_col="plv_mean",
        title="Dyad regime map: PLV (mean over last window)",
        out_path=fig_dir / "dyad_heatmap_plv.png",
    )

    heatmap_from_conditions(
        cond_df,
        value_col="r_mean",
        title="Dyad regime map: global coherence R (mean over last window)",
        out_path=fig_dir / "dyad_heatmap_r.png",
    )

    print(f"made figures in: {fig_dir}")


if __name__ == "__main__":
    main()
