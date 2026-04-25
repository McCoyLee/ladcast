import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


LOWER_IS_BETTER = {
    "mse",
    "rmse",
    "rmse_global",
    "rmse_mean",
    "mae",
    "crps",
    "spread_skill_abs_error",
    "abs_bias",
}
HIGHER_IS_BETTER = {"acc", "spread"}


def load_metric(metric_dir: Path, name: str, required: bool = True) -> Optional[np.ndarray]:
    path = metric_dir / f"{name}.npy"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing metric file: {path}")
        return None
    return np.load(path)


def load_timestamps(metric_dir: Path) -> Optional[np.ndarray]:
    path = metric_dir / "timestamp.npy"
    if not path.exists():
        return None
    return np.load(path)


def scalar_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def by_lead(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 1:
        raise ValueError(f"Expected metric array with lead dimension, got shape={arr.shape}")
    return np.nanmean(arr, axis=tuple(range(arr.ndim - 1)))


def by_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected metric array with channel dimension, got shape={arr.shape}")
    axes = tuple(i for i in range(arr.ndim) if i != 1)
    return np.nanmean(arr, axis=axes)


def per_init(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 1:
        raise ValueError(f"Expected metric array with initialization dimension, got shape={arr.shape}")
    return np.nanmean(arr, axis=tuple(range(1, arr.ndim)))


def align_axis0(
    run: np.ndarray,
    baseline: np.ndarray,
    run_timestamps: Optional[np.ndarray],
    baseline_timestamps: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if run_timestamps is None or baseline_timestamps is None:
        if run.shape != baseline.shape:
            raise ValueError(
                "Metric shapes differ and timestamp.npy is unavailable for alignment: "
                f"run={run.shape}, baseline={baseline.shape}"
            )
        return run, baseline, run_timestamps

    run_index = {int(t): i for i, t in enumerate(run_timestamps)}
    baseline_index = {int(t): i for i, t in enumerate(baseline_timestamps)}
    common = np.asarray(sorted(set(run_index) & set(baseline_index)), dtype=np.int64)
    if common.size == 0:
        raise ValueError("No overlapping timestamps between run and baseline metrics.")

    run_sel = np.asarray([run_index[int(t)] for t in common], dtype=np.int64)
    baseline_sel = np.asarray([baseline_index[int(t)] for t in common], dtype=np.int64)
    return run[run_sel], baseline[baseline_sel], common


def bootstrap_mean_ci(
    delta_per_init: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    delta_per_init = np.asarray(delta_per_init, dtype=np.float64)
    delta_per_init = delta_per_init[np.isfinite(delta_per_init)]
    if delta_per_init.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if delta_per_init.size == 1 or samples <= 0:
        mean = float(np.mean(delta_per_init))
        return {"mean": mean, "ci_low": mean, "ci_high": mean}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta_per_init.size, size=(samples, delta_per_init.size))
    boot = delta_per_init[idx].mean(axis=1)
    return {
        "mean": float(np.mean(delta_per_init)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def direction(metric_name: str) -> str:
    if metric_name in HIGHER_IS_BETTER or metric_name.endswith("_acc"):
        return "higher"
    return "lower"


def skill_score(metric_name: str, run_value: float, baseline_value: float) -> float:
    if not np.isfinite(run_value) or not np.isfinite(baseline_value) or baseline_value == 0:
        return float("nan")
    if direction(metric_name) == "higher":
        return float(run_value / baseline_value - 1.0)
    return float(1.0 - run_value / baseline_value)


def summarize_array(metric_name: str, run: np.ndarray, baseline: np.ndarray, seed: int, bootstrap_samples: int) -> dict:
    delta = run - baseline
    run_mean = scalar_mean(run)
    baseline_mean = scalar_mean(baseline)
    delta_mean = run_mean - baseline_mean
    delta_per_init = per_init(run) - per_init(baseline)
    if direction(metric_name) == "higher":
        improved = delta_per_init > 0
        bootstrap_improvement_probability = float(np.mean(delta_per_init > 0))
    else:
        improved = delta_per_init < 0
        bootstrap_improvement_probability = float(np.mean(delta_per_init < 0))

    stats = bootstrap_mean_ci(
        delta_per_init,
        samples=bootstrap_samples,
        seed=seed,
    )
    stats.update(
        {
            "run_mean": run_mean,
            "baseline_mean": baseline_mean,
            "delta_mean": delta_mean,
            "relative_skill": skill_score(metric_name, run_mean, baseline_mean),
            "win_rate_by_initialization": float(np.mean(improved)),
            "n_initializations": int(delta_per_init.size),
            "improvement_probability_by_initialization": bootstrap_improvement_probability,
            "run_by_lead": by_lead(run).tolist(),
            "baseline_by_lead": by_lead(baseline).tolist(),
            "delta_by_lead": by_lead(delta).tolist(),
            "run_by_channel": by_channel(run).tolist(),
            "baseline_by_channel": by_channel(baseline).tolist(),
            "delta_by_channel": by_channel(delta).tolist(),
            "direction": direction(metric_name),
        }
    )
    return stats


def metric_arrays(metric_dir: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    ens_mse = load_metric(metric_dir, "ens_mse")
    out["mse"] = ens_mse
    out["rmse"] = np.sqrt(ens_mse)
    # Global RMSE matches eval_ladcast_latent_predictions.py summary:
    # sqrt(mean(MSE)), computed per init/channel/lead before averaging.
    out["rmse_global"] = np.sqrt(ens_mse)

    optional_files = {
        "mae": "mae",
        "crps": "crps",
        "acc": "ens_acc",
        "spread": "ensemble_spread",
        "abs_bias": "bias",
    }
    for metric_name, file_name in optional_files.items():
        arr = load_metric(metric_dir, file_name, required=False)
        if arr is None:
            continue
        if metric_name == "abs_bias":
            arr = np.abs(arr)
        out[metric_name] = arr
    spread_skill_ratio = load_metric(metric_dir, "spread_skill_ratio", required=False)
    if spread_skill_ratio is not None:
        out["spread_skill_abs_error"] = np.abs(spread_skill_ratio - 1.0)
    return out


def summarize(metric_dir: Path) -> dict:
    arrays = metric_arrays(metric_dir)
    ens_mse = arrays["mse"]
    summary = {
        "mse_mean": scalar_mean(ens_mse),
        "rmse_mean": scalar_mean(arrays["rmse"]),
        "rmse_global": float(np.sqrt(np.nanmean(ens_mse))),
        "rmse_by_lead": by_lead(arrays["rmse"]).tolist(),
        "rmse_global_by_lead": np.sqrt(by_lead(ens_mse)).tolist(),
    }
    for name, arr in arrays.items():
        if name in {"mse", "rmse", "rmse_global"}:
            continue
        summary[f"{name}_mean"] = scalar_mean(arr)
        summary[f"{name}_by_lead"] = by_lead(arr).tolist()
    return summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of a LaDCast experiment metric directory against a matched baseline."
    )
    parser.add_argument("--run_metrics", required=True, type=Path)
    parser.add_argument("--baseline_metrics", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--step_size_hour", type=int, default=6)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output_csv_prefix",
        type=Path,
        default=None,
        help="Prefix for by-lead/by-channel/per-init CSV files. Defaults to output_json without .json.",
    )
    args = parser.parse_args()

    run_timestamps = load_timestamps(args.run_metrics)
    baseline_timestamps = load_timestamps(args.baseline_metrics)
    run_arrays = metric_arrays(args.run_metrics)
    baseline_arrays = metric_arrays(args.baseline_metrics)

    metric_reports = {}
    per_lead_rows = []
    per_channel_rows = []
    per_init_rows = []
    aligned_timestamps = None

    for metric_name in sorted(set(run_arrays) & set(baseline_arrays)):
        run_arr, baseline_arr, aligned_timestamps = align_axis0(
            run_arrays[metric_name],
            baseline_arrays[metric_name],
            run_timestamps,
            baseline_timestamps,
        )

        # For rmse_global, summarize from MSE at the aggregate level while retaining
        # per-init paired deltas as sqrt(mean MSE per init).
        if metric_name == "rmse_global":
            run_mse, baseline_mse, _ = align_axis0(
                run_arrays["mse"],
                baseline_arrays["mse"],
                run_timestamps,
                baseline_timestamps,
            )
            run_scalar = np.sqrt(np.nanmean(run_mse))
            baseline_scalar = np.sqrt(np.nanmean(baseline_mse))
            delta_per_init = np.sqrt(np.nanmean(run_mse, axis=(1, 2))) - np.sqrt(
                np.nanmean(baseline_mse, axis=(1, 2))
            )
            stats = bootstrap_mean_ci(
                delta_per_init,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            stats.update(
                {
                    "run_mean": float(run_scalar),
                    "baseline_mean": float(baseline_scalar),
                    "delta_mean": float(run_scalar - baseline_scalar),
                    "relative_skill": skill_score(metric_name, float(run_scalar), float(baseline_scalar)),
                    "win_rate_by_initialization": float(np.mean(delta_per_init < 0)),
                    "n_initializations": int(delta_per_init.size),
                    "improvement_probability_by_initialization": float(np.mean(delta_per_init < 0)),
                    "run_by_lead": np.sqrt(by_lead(run_mse)).tolist(),
                    "baseline_by_lead": np.sqrt(by_lead(baseline_mse)).tolist(),
                    "delta_by_lead": (np.sqrt(by_lead(run_mse)) - np.sqrt(by_lead(baseline_mse))).tolist(),
                    "run_by_channel": np.sqrt(by_channel(run_mse)).tolist(),
                    "baseline_by_channel": np.sqrt(by_channel(baseline_mse)).tolist(),
                    "delta_by_channel": (np.sqrt(by_channel(run_mse)) - np.sqrt(by_channel(baseline_mse))).tolist(),
                    "direction": "lower",
                }
            )
        else:
            stats = summarize_array(
                metric_name,
                run_arr,
                baseline_arr,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
        metric_reports[metric_name] = stats

        for lead_idx, (run_v, base_v, delta_v) in enumerate(
            zip(stats["run_by_lead"], stats["baseline_by_lead"], stats["delta_by_lead"])
        ):
            per_lead_rows.append(
                {
                    "metric": metric_name,
                    "lead_index": lead_idx + 1,
                    "lead_hour": (lead_idx + 1) * args.step_size_hour,
                    "run": run_v,
                    "baseline": base_v,
                    "delta": delta_v,
                    "relative_skill": skill_score(metric_name, run_v, base_v),
                }
            )
        for channel_idx, (run_v, base_v, delta_v) in enumerate(
            zip(stats["run_by_channel"], stats["baseline_by_channel"], stats["delta_by_channel"])
        ):
            per_channel_rows.append(
                {
                    "metric": metric_name,
                    "channel": channel_idx,
                    "run": run_v,
                    "baseline": base_v,
                    "delta": delta_v,
                    "relative_skill": skill_score(metric_name, run_v, base_v),
                }
            )
        for init_idx, delta_v in enumerate(per_init(run_arr) - per_init(baseline_arr)):
            row = {
                "metric": metric_name,
                "initialization_index": init_idx,
                "delta": float(delta_v),
            }
            if aligned_timestamps is not None:
                row["timestamp"] = int(aligned_timestamps[init_idx])
            per_init_rows.append(row)

    delta = {}
    for key in sorted(set(summarize(args.run_metrics)) & set(summarize(args.baseline_metrics))):
        if key.endswith("_mean") or key == "rmse_global":
            delta[key] = summarize(args.run_metrics)[key] - summarize(args.baseline_metrics)[key]

    csv_prefix = args.output_csv_prefix
    if csv_prefix is None:
        csv_prefix = args.output_json.with_suffix("")
    write_csv(csv_prefix.parent / f"{csv_prefix.name}_by_lead.csv", per_lead_rows)
    write_csv(csv_prefix.parent / f"{csv_prefix.name}_by_channel.csv", per_channel_rows)
    write_csv(csv_prefix.parent / f"{csv_prefix.name}_per_init.csv", per_init_rows)

    report = {
        "run_metrics": str(args.run_metrics),
        "baseline_metrics": str(args.baseline_metrics),
        "aligned_timestamps": None if aligned_timestamps is None else aligned_timestamps.tolist(),
        "run": summarize(args.run_metrics),
        "baseline": summarize(args.baseline_metrics),
        "delta_run_minus_baseline": delta,
        "paired_metric_reports": metric_reports,
        "csv_outputs": {
            "by_lead": str(csv_prefix.parent / f"{csv_prefix.name}_by_lead.csv"),
            "by_channel": str(csv_prefix.parent / f"{csv_prefix.name}_by_channel.csv"),
            "per_init": str(csv_prefix.parent / f"{csv_prefix.name}_per_init.csv"),
        },
        "interpretation": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
            "relative_skill": "positive means the run improves over baseline",
            "ci_low/ci_high": "paired bootstrap 95% CI over initialization times",
            "rmse_mean": "mean(sqrt(MSE)) over initialization/channel/lead; kept for backward compatibility",
            "rmse_global": "sqrt(mean(MSE)); matches eval_ladcast_latent_predictions.py summary",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "delta_run_minus_baseline": delta,
                "relative_skill": {
                    name: stats["relative_skill"] for name, stats in metric_reports.items()
                },
                "bootstrap_ci": {
                    name: {
                        "mean": stats["mean"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                    }
                    for name, stats in metric_reports.items()
                },
                "saved": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
