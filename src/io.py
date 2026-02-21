# src/io.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def utc_timestamp_slug() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d_%H%M%S_utc")


def make_run_dir(base_dir: str = "outputs", tag: str = "run") -> Path:
    run_dir = Path(base_dir) / f"{utc_timestamp_slug()}_{tag}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_theta_csv(time: np.ndarray, theta: np.ndarray, path: str | Path) -> None:
    df = pd.DataFrame(theta, columns=[f"theta_{i}" for i in range(theta.shape[1])])
    df.insert(0, "time", time)
    df.to_csv(path, index=False)


def save_metrics_timeseries(time: np.ndarray, metrics: dict[str, np.ndarray], path: str | Path) -> None:
    df = pd.DataFrame({k: v for k, v in metrics.items()})
    df.insert(0, "time", time)
    df.to_csv(path, index=False)


def save_metrics_summary(summary: dict, path: str | Path) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)
