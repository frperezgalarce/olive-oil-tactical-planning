#!/usr/bin/env python
"""Global sensitivity analysis for olive biomass/yield estimation.

This script wraps the existing biomass model in ``src/biomassmodel/yield_estimation.py``
without changing model behavior. It samples numeric model parameters independently,
runs Monte Carlo simulations, and computes parameter impact rankings.
"""

from __future__ import annotations
import argparse
import ast
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global sensitivity analysis for biomass/yield model")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of random parameter sets")
    parser.add_argument("--seed", type=int, default=1952, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs") / "sensitivity",
        help="Base output directory (run subfolder is created automatically)",
    )
    parser.add_argument(
        "--method",
        choices=["corr", "rf", "all"],
        default="all",
        help="Impact methods to run",
    )
    parser.add_argument(
        "--weather_csv",
        type=Path,
        default=None,
        help="Optional weather CSV path. Defaults to data/data_20150630_to_20250630.csv if present.",
    )
    parser.add_argument(
        "--from_samples_csv",
        type=Path,
        default=None,
        help="Optional existing samples.csv. If provided, skips model runs and regenerates outputs from this table.",
    )
    parser.add_argument(
        "--reuse_run_dir",
        type=Path,
        default=None,
        help="Optional existing run directory to overwrite ranking/plot/metadata outputs when using --from_samples_csv.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prepare_weather_dataframe(weather_csv: Path) -> pd.DataFrame:
    """Load and derive weather variables expected by ``get_estimation``.

    Expected input columns include: DATE, T2M_MAX, T2M_MIN, ALLSKY_SFC_SW_DWN,
    RH2M, PRECTOTCORR.
    """
    df = pd.read_csv(weather_csv)
    if "DATE" not in df.columns:
        raise ValueError(f"Missing DATE column in weather file: {weather_csv}")

    required_raw = ["T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "RH2M", "PRECTOTCORR"]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required weather columns: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").set_index("DATE")

    # NASA POWER ALLSKY_SFC_SW_DWN is commonly MJ/m2/day; keep rad_factor=1.0.
    rad_factor = 1.0
    df["SRAD_MJ"] = df["ALLSKY_SFC_SW_DWN"].astype(float) * rad_factor
    df["PAR_MJ"] = df["SRAD_MJ"] * 0.48
    df["TAVG"] = (df["T2M_MAX"].astype(float) + df["T2M_MIN"].astype(float)) / 2.0

    es = 0.6108 * np.exp(17.27 * df["TAVG"] / (df["TAVG"] + 237.3))
    ea = es * (df["RH2M"].astype(float) / 100.0)
    df["VPD_kPa"] = np.maximum(0.1, es - ea)
    df["RAIN_mm"] = df["PRECTOTCORR"].astype(float)

    required_model = ["TAVG", "PAR_MJ", "RAIN_mm", "VPD_kPa", "SRAD_MJ", "T2M_MAX"]
    missing_model = [c for c in required_model if c not in df.columns]
    if missing_model:
        raise ValueError(f"Missing derived model columns: {missing_model}")

    return df


def default_weather_csv(repo_root: Path) -> Path:
    preferred = repo_root / "data" / "data_20150630_to_20250630.csv"
    if preferred.exists():
        return preferred

    candidates = sorted((repo_root / "data").glob("data_*_to_*.csv"))
    if not candidates:
        raise FileNotFoundError("No weather CSV found under data/")
    return candidates[0]


def load_base_params(params_path: Path) -> Dict[str, object]:
    with params_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_used_param_keys_from_source(yield_file: Path) -> List[str]:
    """Parse ``yield_estimation.py`` and extract keys accessed as params["..."]"""
    source = yield_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    keys = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        if not isinstance(node.value, ast.Name) or node.value.id != "params":
            continue

        # Python 3.9+: slice is directly expr
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            keys.add(slice_node.value)

    return sorted(keys)


def get_numeric_model_params(base_params: Dict[str, object], used_keys: List[str]) -> Tuple[List[str], Dict[str, float]]:
    numeric_params: Dict[str, float] = {}
    for key in used_keys:
        if key not in base_params:
            continue
        value = base_params[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_params[key] = float(value)

    return sorted(numeric_params.keys()), numeric_params


def sample_parameter_matrix(
    param_names: List[str],
    base_values: Dict[str, float],
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, len(param_names)), dtype=float)
    bounds: Dict[str, Tuple[float, float]] = {}

    for j, name in enumerate(param_names):
        v = base_values[name]
        if v == 0.0:
            # Zero-centered parameters cannot use +/-20% of zero; use a small symmetric range.
            lo, hi = -0.2, 0.2
        else:
            lo = 0.8 * v
            hi = 1.2 * v
            lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        x[:, j] = rng.uniform(lo, hi, size=n_samples)
        bounds[name] = (float(lo), float(hi))

    return x, bounds


def fit_standardized_linear_coeffs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_std = x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    z_x = (x - x.mean(axis=0)) / x_std

    y_std = float(y.std())
    if y_std == 0.0:
        return np.zeros(x.shape[1], dtype=float)
    z_y = (y - y.mean()) / y_std

    beta, *_ = np.linalg.lstsq(z_x, z_y, rcond=None)
    return np.abs(beta)


def compute_rf_metrics(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for RF/permutation importance") from exc

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    rf_importance = model.feature_importances_

    perm = permutation_importance(
        model,
        x,
        y,
        n_repeats=15,
        random_state=seed,
        n_jobs=-1,
    )
    return rf_importance, perm.importances_mean


def plot_correlation_bars(rank_df: pd.DataFrame, outpath: Path) -> None:
    df = rank_df.copy()
    df["abs_pearson"] = df["pearson_r"].abs()
    df = df.sort_values("abs_pearson", ascending=True)

    fig_h = max(6, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in df["pearson_r"]]
    ax.barh(df["param"], df["pearson_r"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Pearson correlation with target")
    ax.set_ylabel("Parameter")
    ax.set_title("Parameter Correlation with Target")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_permutation_importance(rank_df: pd.DataFrame, outpath: Path) -> None:
    df = rank_df.dropna(subset=["perm_importance_mean"]).copy()
    df = df.sort_values("perm_importance_mean", ascending=True)

    fig_h = max(6, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(df["param"], df["perm_importance_mean"], color="#1b9e77")
    ax.set_xlabel("Permutation importance (mean)")
    ax.set_ylabel("Parameter")
    ax.set_title("Permutation Importance Ranking")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_top_scatter_grid(samples_df: pd.DataFrame, rank_df: pd.DataFrame, target_col: str, outpath: Path) -> None:
    top_params = rank_df.assign(abs_pearson=rank_df["pearson_r"].abs()).sort_values(
        "abs_pearson", ascending=False
    )["param"].head(6)

    if top_params.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, param in enumerate(top_params):
        ax = axes[idx]
        ax.scatter(samples_df[param], samples_df[target_col], s=10, alpha=0.35, color="#2c7fb8")
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel(target_col)

    for j in range(len(top_params), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def evaluate_target(
    weather_df: pd.DataFrame,
    base_params: Dict[str, object],
    param_names: List[str],
    x: np.ndarray,
    suppress_model_stdout: bool = True,
) -> Tuple[np.ndarray, int]:
    """Run model for each sampled parameter set and return target vector."""
    from yield_estimation import get_estimation

    y = np.full(x.shape[0], np.nan, dtype=float)
    failures = 0

    for i in range(x.shape[0]):
        params_i = dict(base_params)
        for j, p in enumerate(param_names):
            params_i[p] = float(x[i, j])

        try:
            if suppress_model_stdout:
                with redirect_stdout(io.StringIO()):
                    target = get_estimation(weather_df, params_i, VERBOSE=False)
            else:
                target = get_estimation(weather_df, params_i, VERBOSE=False)
            y[i] = float(target)
        except Exception:
            failures += 1
            y[i] = np.nan

        if (i + 1) % max(1, x.shape[0] // 10) == 0:
            print(f"Progress: {i + 1}/{x.shape[0]} samples evaluated")

    return y, failures


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    biomass_dir = repo_root / "src" / "biomassmodel"
    if str(biomass_dir) not in sys.path:
        sys.path.insert(0, str(biomass_dir))

    params_path = biomass_dir / "params.json"
    yield_file = biomass_dir / "yield_estimation.py"

    weather_csv = args.weather_csv if args.weather_csv is not None else default_weather_csv(repo_root)
    if not weather_csv.is_absolute():
        weather_csv = (repo_root / weather_csv).resolve()

    out_base = args.outdir if args.outdir.is_absolute() else (repo_root / args.outdir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.reuse_run_dir is not None:
        run_dir = args.reuse_run_dir if args.reuse_run_dir.is_absolute() else (repo_root / args.reuse_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = out_base / f"run_{run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

    base_params = load_base_params(params_path)
    used_keys = extract_used_param_keys_from_source(yield_file)
    param_names, numeric_base = get_numeric_model_params(base_params, used_keys)

    if not param_names:
        raise RuntimeError("No numeric model parameters found from params.json used by yield_estimation.py")

    target_col = "target_avg_yield_fresh_kg_ha"
    if args.from_samples_csv is not None:
        samples_path = args.from_samples_csv if args.from_samples_csv.is_absolute() else (repo_root / args.from_samples_csv)
        samples_df = pd.read_csv(samples_path)
        missing_cols = [c for c in param_names + [target_col] if c not in samples_df.columns]
        if missing_cols:
            raise RuntimeError(f"Provided samples CSV is missing required columns: {missing_cols}")
        failures = int(samples_df[target_col].isna().sum())
        args.n_samples = int(samples_df.shape[0])
        bounds = {}
        for p in param_names:
            bounds[p] = (float(samples_df[p].min()), float(samples_df[p].max()))
    else:
        weather_df = prepare_weather_dataframe(weather_csv)
        x, bounds = sample_parameter_matrix(
            param_names=param_names,
            base_values=numeric_base,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        y, failures = evaluate_target(weather_df, base_params, param_names, x)
        samples_df = pd.DataFrame(x, columns=param_names)
        samples_df[target_col] = y

    samples_csv = run_dir / "samples.csv"
    samples_df.to_csv(samples_csv, index=False)

    valid = samples_df.dropna(subset=[target_col]).copy()
    if valid.empty:
        raise RuntimeError(
            "No valid target outputs were produced. "
            "Model may require additional inputs or parameter constraints."
        )

    x_valid = valid[param_names].to_numpy(dtype=float)
    y_valid = valid[target_col].to_numpy(dtype=float)

    pearson = valid[param_names].corrwith(valid[target_col], method="pearson")
    spearman = valid[param_names].corrwith(valid[target_col], method="spearman")
    std_beta_abs = fit_standardized_linear_coeffs(x_valid, y_valid)

    rf_importance = np.full(len(param_names), np.nan, dtype=float)
    perm_importance = np.full(len(param_names), np.nan, dtype=float)

    if args.method in {"rf", "all"}:
        try:
            rf_importance, perm_importance = compute_rf_metrics(x_valid, y_valid, args.seed)
        except RuntimeError as exc:
            print(f"Warning: {exc}. RF-based metrics will be NaN.")

    rank_df = pd.DataFrame(
        {
            "param": param_names,
            "base_value": [numeric_base[p] for p in param_names],
            "min_sampled": [bounds[p][0] for p in param_names],
            "max_sampled": [bounds[p][1] for p in param_names],
            "pearson_r": [float(pearson[p]) for p in param_names],
            "spearman_r": [float(spearman[p]) for p in param_names],
            "std_beta_abs": std_beta_abs,
            "rf_importance": rf_importance,
            "perm_importance_mean": perm_importance,
        }
    )

    rank_df = rank_df.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    ranking_csv = run_dir / "sensitivity_rankings.csv"
    rank_df.to_csv(ranking_csv, index=False)

    correlation_png = run_dir / "correlation_plot.png"
    plot_correlation_bars(rank_df, correlation_png)

    perm_png = run_dir / "permutation_importance.png"
    if rank_df["perm_importance_mean"].notna().any():
        plot_permutation_importance(rank_df, perm_png)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Permutation importance unavailable", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(perm_png, dpi=150)
        plt.close(fig)

    scatter_png = run_dir / "top6_scatter_grid.png"
    plot_top_scatter_grid(valid, rank_df, target_col, scatter_png)

    metadata = {
        "timestamp": run_stamp,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "method": args.method,
        "from_samples_csv": str(args.from_samples_csv) if args.from_samples_csv is not None else None,
        "target_column": target_col,
        "weather_csv": str(weather_csv),
        "params_json": str(params_path),
        "yield_source": str(yield_file),
        "n_model_numeric_params": len(param_names),
        "n_valid_outputs": int(valid.shape[0]),
        "n_failed_outputs": int(failures),
        "git_commit": git_commit_hash(repo_root),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    top = rank_df.head(10).copy()
    top["abs_pearson"] = top["pearson_r"].abs()

    print("\nSensitivity analysis complete")
    print(f"Run directory: {run_dir}")
    print(f"Samples evaluated: {args.n_samples} | Valid: {valid.shape[0]} | Failed: {failures}")
    print(f"Model parameters varied: {len(param_names)}")
    print("Top parameters by |pearson_r|:")
    for _, row in top.iterrows():
        print(f"  - {row['param']}: pearson={row['pearson_r']:.4f}, spearman={row['spearman_r']:.4f}")


if __name__ == "__main__":
    main()
