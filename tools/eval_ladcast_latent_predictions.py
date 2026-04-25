import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_yyyymmddhh(path: Path) -> pd.Timestamp:
    stem = path.stem
    value = stem.split("_")[-1]
    return pd.to_datetime(value, format="%Y%m%d%H")


def load_latent_dataset(path: str, var_name: str) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=False)
    if var_name not in ds:
        if var_name == "latent" and "latents" in ds:
            var_name = "latents"
        elif var_name == "latents" and "latent" in ds:
            var_name = "latent"
        else:
            raise KeyError(f"{var_name!r} not found in {path}. Variables: {list(ds.data_vars)}")
    arr = ds[var_name]
    rename = {}
    for old, new in {"channel": "C", "lat": "H", "lon": "W"}.items():
        if old in arr.dims:
            rename[old] = new
    if rename:
        arr = arr.rename(rename)
    return arr.transpose("time", "C", "H", "W")


def crps_by_channel_lead(members: np.ndarray, target: np.ndarray) -> np.ndarray:
    # members: (E, C, T, H, W), target: (C, T, H, W)
    term1 = np.mean(np.abs(members - target[None, ...]), axis=(0, 3, 4))
    ens = members.shape[0]
    if ens < 2:
        return term1
    pair_sum = np.zeros_like(term1)
    n_pairs = 0
    for i in range(ens):
        for j in range(i + 1, ens):
            pair_sum += np.mean(np.abs(members[i] - members[j]), axis=(2, 3))
            n_pairs += 1
    return term1 - 0.5 * pair_sum / n_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent_YYYYMMDDHH.npy predictions against latent zarr truth.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--latent_path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--var_name", default="latent")
    parser.add_argument("--step_size_hour", type=int, default=6)
    parser.add_argument("--total_lead_time_hour", type=int, default=240)
    parser.add_argument("--crop_init", action="store_true")
    parser.add_argument("--force_ens_size", type=int, default=None)
    args = parser.parse_args()

    latent = load_latent_dataset(args.latent_path, args.var_name)
    pred_paths = sorted(args.pred_dir.glob("latent_*.npy"))
    if not pred_paths:
        raise FileNotFoundError(f"No latent_*.npy files found in {args.pred_dir}")

    total_steps = args.total_lead_time_hour // args.step_size_hour
    timestamps = []
    mse_all = []
    mae_all = []
    crps_all = []
    spread_all = []
    spread_skill_ratio_all = []
    bias_all = []

    for path in pred_paths:
        init_time = parse_yyyymmddhh(path)
        pred = np.load(path).astype(np.float32)  # (E, C, T, H, W)
        if args.crop_init:
            pred = pred[:, :, 1:, :, :]
        if args.force_ens_size is not None:
            pred = pred[: args.force_ens_size]
        pred = pred[:, :, :total_steps, :, :]
        lead_times = pd.date_range(
            init_time + pd.Timedelta(hours=args.step_size_hour),
            periods=pred.shape[2],
            freq=f"{args.step_size_hour}h",
        )
        truth = latent.sel(time=lead_times).values.astype(np.float32)  # (T, C, H, W)
        truth = np.transpose(truth, (1, 0, 2, 3))  # (C, T, H, W)
        if pred.shape[1:] != truth.shape:
            raise ValueError(f"Shape mismatch for {path}: pred={pred.shape}, truth={truth.shape}")

        ens_mean = pred.mean(axis=0)
        mse = np.mean((ens_mean - truth) ** 2, axis=(2, 3))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ens_mean - truth), axis=(2, 3))
        bias = np.mean(ens_mean - truth, axis=(2, 3))
        spread = np.sqrt(np.mean((pred - ens_mean[None, ...]) ** 2, axis=(0, 3, 4)))
        spread_skill_ratio = spread / np.maximum(rmse, 1e-12)

        mse_all.append(mse)
        mae_all.append(mae)
        bias_all.append(bias)
        spread_all.append(spread)
        spread_skill_ratio_all.append(spread_skill_ratio)
        crps_all.append(crps_by_channel_lead(pred, truth))
        timestamps.append(int(init_time.strftime("%Y%m%d%H")))

    mse_stack = np.stack(mse_all, axis=0)
    mae_stack = np.stack(mae_all, axis=0)
    crps_stack = np.stack(crps_all, axis=0)
    spread_stack = np.stack(spread_all, axis=0)
    spread_skill_ratio_stack = np.stack(spread_skill_ratio_all, axis=0)
    bias_stack = np.stack(bias_all, axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "timestamp.npy", np.asarray(timestamps, dtype=np.int64))
    np.save(args.output / "ens_mse.npy", mse_stack)
    np.save(args.output / "mae.npy", mae_stack)
    np.save(args.output / "crps.npy", crps_stack)
    np.save(args.output / "ensemble_spread.npy", spread_stack)
    np.save(args.output / "spread_skill_ratio.npy", spread_skill_ratio_stack)
    np.save(args.output / "bias.npy", bias_stack)
    summary = {
        "pred_dir": str(args.pred_dir),
        "latent_path": args.latent_path,
        "n_initializations": len(timestamps),
        "rmse_mean": float(np.sqrt(np.nanmean(mse_stack))),
        "rmse_gridpoint_mean": float(np.nanmean(np.sqrt(mse_stack))),
        "mae_mean": float(np.nanmean(mae_stack)),
        "crps_mean": float(np.nanmean(crps_stack)),
        "ensemble_spread_mean": float(np.nanmean(spread_stack)),
        "spread_skill_ratio_mean": float(np.nanmean(spread_skill_ratio_stack)),
        "abs_bias_mean": float(np.nanmean(np.abs(bias_stack))),
        "bias_mean": float(np.nanmean(bias_stack)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
