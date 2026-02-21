# scripts/make_figures_classroom_sweep.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_classroom_sweep(outputs_dir: Path) -> Path:
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and "classroom_sweep" in p.name]
    if not candidates:
        raise FileNotFoundError("No classroom_sweep folders found in outputs/")
    return sorted(candidates)[-1]


def heatmap(df: pd.DataFrame, value: str, title: str, out_path: Path) -> None:
    x = np.sort(df["k_ts"].unique())
    y = np.sort(df["k_st"].unique())

    pivot = df.pivot(index="k_st", columns="k_ts", values=value).reindex(index=y, columns=x)

    plt.figure(figsize=(7, 4.8))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(x)), [str(v) for v in x], rotation=45)
    plt.yticks(np.arange(len(y)), [str(v) for v in y])
    plt.xlabel("k_ts (teacher→student)")
    plt.ylabel("k_st (student→teacher)")
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
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_classroom_sweep(outputs)

    df = pd.read_csv(sweep_dir / "classroom_sweep_conditions.csv")

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # one panel per k_ss
    for k_ss in sorted(df["k_ss"].unique()):
        sub = df[df["k_ss"] == k_ss].copy()

        heatmap(
            sub,
            value="ts_plv_mean",
            title=f"Classroom sweep (k_ss={k_ss}): teacher–student PLV mean",
            out_path=fig_dir / f"classroom_kss{k_ss}_ts_plv_mean.png",
        )
        heatmap(
            sub,
            value="teacher_dom_mean",
            title=f"Classroom sweep (k_ss={k_ss}): teacher dominance mean",
            out_path=fig_dir / f"classroom_kss{k_ss}_teacher_dom_mean.png",
        )
        heatmap(
            sub,
            value="r_mean",
            title=f"Classroom sweep (k_ss={k_ss}): global coherence R mean",
            out_path=fig_dir / f"classroom_kss{k_ss}_r_mean.png",
        )

        heatmap(
            sub,
            value="teacher_antiphase_dom_mean",
            title=f"Classroom sweep (k_ss={k_ss}): teacher antiphase dominance mean",
            out_path=fig_dir / f"classroom_kss{k_ss}_teacher_antiphase_dom_mean.png",
        )


    print(f"made classroom sweep diagnostics in: {fig_dir}")


if __name__ == "__main__":
    main()
