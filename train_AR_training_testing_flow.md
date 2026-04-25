# LaDCast RouteB 训练、推理、评估与公平对比流程

下面命令按顺序复制执行即可。默认使用 `checkpoint-60000`，因为当前 `your_config.yaml` 和 `your_config_no_terrain.yaml` 已设置为 6 小时训练网格、`return_seq_len=4`、`num_training_steps=60000`。

## 0. 通用环境变量

```bash
source /data/user_envs/limaocheng/ladcast/bin/activate
cd /home/limaocheng/ladcast

export PY=/data/user_envs/limaocheng/ladcast/bin/python3.11
export CKPT=60000
export DATA_PATH=/home/limaocheng/data/ERA5_ladcast_routeB_1979_2024.zarr
export LATENT_PATH=/home/limaocheng/ladcast/data/routeB_latent_train.zarr
export LATENT_NORM=/home/limaocheng/ladcast/static/ERA5_routeB_latent_normal_1979_2017.json
export PHYS_NORM=/home/limaocheng/ladcast/static/ERA5_routeB_normal_1979_2017.json
export LSM=/home/limaocheng/ladcast/ladcast/static/240x121_land_sea_mask.pt
export ORO=/home/limaocheng/ladcast/ladcast/static/240x121_orography.pt
export AE=/home/limaocheng/ladcast/checkpoints/routeB_ae_hf
export TEST_START=2018-01-01T00:00:00
export TEST_END=2018-02-01T12:00:00
```

## 1. 训练 matched no-terrain baseline

```bash
mkdir -p output/ladcast_no_terrain

CUDA_VISIBLE_DEVICES=2 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  ladcast/train_AR.py \
  --config your_config_no_terrain.yaml \
  --ar_cls transformer \
  --encdec_cls dcae \
  --gradient_checkpointing \
  --checkpoints_total_limit 8 \
  --latent_norm_json_path "$LATENT_NORM" \
  --skip_encdec
```

## 2. 训练 terrain 模型

```bash
mkdir -p output/ladcast_terrain

CUDA_VISIBLE_DEVICES=2 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  ladcast/train_AR.py \
  --config your_config.yaml \
  --ar_cls transformer \
  --use_terrain \
  --terrain_out_channels 8 \
  --terrain_alpha_init 0.0 \
  --terrain_output_gate_init 0.0 \
  --terrain_lr_scale 2.0 \
  --gradient_checkpointing \
  --checkpoints_total_limit 8 \
  --latent_norm_json_path "$LATENT_NORM" \
  --skip_encdec
```

## 3. 训练参数量匹配 zero-terrain control

这组保留 terrain 分支参数，但训练和验证时把 terrain feature 置零。它用于证明收益来自地形信息，而不是来自额外参数量。

```bash
sed \
  -e 's#output/ladcast_terrain#output/ladcast_terrain_zero#g' \
  -e 's#ladcast-terrain#ladcast-terrain-zero#g' \
  your_config.yaml > your_config_terrain_zero.yaml

mkdir -p output/ladcast_terrain_zero

CUDA_VISIBLE_DEVICES=2 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  ladcast/train_AR.py \
  --config your_config_terrain_zero.yaml \
  --ar_cls transformer \
  --use_terrain \
  --zero_terrain_features \
  --terrain_out_channels 8 \
  --terrain_alpha_init 0.0 \
  --terrain_output_gate_init 0.0 \
  --terrain_lr_scale 2.0 \
  --gradient_checkpointing \
  --checkpoints_total_limit 8 \
  --latent_norm_json_path "$LATENT_NORM" \
  --skip_encdec
```

## 4. 生成 no-terrain baseline latent 预测

```bash
rm -rf "predictions/ladcast_no_terrain_ckpt${CKPT}_latent"
mkdir -p "predictions/ladcast_no_terrain_ckpt${CKPT}_latent"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  ladcast/evaluate/pred_rollout.py \
  --ar_model_path "/home/limaocheng/ladcast/output/ladcast_no_terrain/checkpoint-${CKPT}/ar_model" \
  --ar_cls transformer \
  --data_path "$DATA_PATH" \
  --latent_normal_json "$LATENT_NORM" \
  --normalization_json "$PHYS_NORM" \
  --lsm_path "$LSM" \
  --orography_path "$ORO" \
  --encdec_model_path "$AE" \
  --start_date "$TEST_START" \
  --end_date "$TEST_END" \
  --ensemble_size 10 \
  --num_inference_steps 20 \
  --return_seq_len 4 \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --input_seq_len 1 \
  --num_samples_per_month 10 \
  --save_as_latent \
  --sampler_type edm \
  --output "/home/limaocheng/ladcast/predictions/ladcast_no_terrain_ckpt${CKPT}_latent"
```

## 5. 生成 terrain latent 预测

```bash
rm -rf "predictions/ladcast_terrain_ckpt${CKPT}_latent"
mkdir -p "predictions/ladcast_terrain_ckpt${CKPT}_latent"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  ladcast/evaluate/pred_rollout.py \
  --ar_model_path "/home/limaocheng/ladcast/output/ladcast_terrain/checkpoint-${CKPT}/ar_model" \
  --terrain_encoder_path "/home/limaocheng/ladcast/output/ladcast_terrain/checkpoint-${CKPT}/terrain_encoder.pt" \
  --terrain_latent_nlat 32 \
  --terrain_latent_nlon 60 \
  --pad_input_height_to 128 \
  --ar_cls transformer \
  --data_path "$DATA_PATH" \
  --latent_normal_json "$LATENT_NORM" \
  --normalization_json "$PHYS_NORM" \
  --lsm_path "$LSM" \
  --orography_path "$ORO" \
  --encdec_model_path "$AE" \
  --start_date "$TEST_START" \
  --end_date "$TEST_END" \
  --ensemble_size 10 \
  --num_inference_steps 20 \
  --return_seq_len 4 \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --input_seq_len 1 \
  --num_samples_per_month 10 \
  --save_as_latent \
  --sampler_type edm \
  --output "/home/limaocheng/ladcast/predictions/ladcast_terrain_ckpt${CKPT}_latent"
```

## 6. 生成 zero-terrain control latent 预测

```bash
rm -rf "predictions/ladcast_terrain_zero_ckpt${CKPT}_latent"
mkdir -p "predictions/ladcast_terrain_zero_ckpt${CKPT}_latent"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  ladcast/evaluate/pred_rollout.py \
  --ar_model_path "/home/limaocheng/ladcast/output/ladcast_terrain_zero/checkpoint-${CKPT}/ar_model" \
  --terrain_encoder_path "/home/limaocheng/ladcast/output/ladcast_terrain_zero/checkpoint-${CKPT}/terrain_encoder.pt" \
  --zero_terrain_features \
  --terrain_latent_nlat 32 \
  --terrain_latent_nlon 60 \
  --pad_input_height_to 128 \
  --ar_cls transformer \
  --data_path "$DATA_PATH" \
  --latent_normal_json "$LATENT_NORM" \
  --normalization_json "$PHYS_NORM" \
  --lsm_path "$LSM" \
  --orography_path "$ORO" \
  --encdec_model_path "$AE" \
  --start_date "$TEST_START" \
  --end_date "$TEST_END" \
  --ensemble_size 10 \
  --num_inference_steps 20 \
  --return_seq_len 4 \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --input_seq_len 1 \
  --num_samples_per_month 10 \
  --save_as_latent \
  --sampler_type edm \
  --output "/home/limaocheng/ladcast/predictions/ladcast_terrain_zero_ckpt${CKPT}_latent"
```

## 7. 可选：同一 terrain checkpoint 的推理期置零 ablation

这组不重新训练，只用 terrain checkpoint，把推理时 terrain feature 置零。它用于判断已训练 terrain 模型是否真的在推理中使用地形条件。

```bash
rm -rf "predictions/ladcast_terrain_ckpt${CKPT}_zero_infer_latent"
mkdir -p "predictions/ladcast_terrain_ckpt${CKPT}_zero_infer_latent"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  ladcast/evaluate/pred_rollout.py \
  --ar_model_path "/home/limaocheng/ladcast/output/ladcast_terrain/checkpoint-${CKPT}/ar_model" \
  --terrain_encoder_path "/home/limaocheng/ladcast/output/ladcast_terrain/checkpoint-${CKPT}/terrain_encoder.pt" \
  --zero_terrain_features \
  --terrain_latent_nlat 32 \
  --terrain_latent_nlon 60 \
  --pad_input_height_to 128 \
  --ar_cls transformer \
  --data_path "$DATA_PATH" \
  --latent_normal_json "$LATENT_NORM" \
  --normalization_json "$PHYS_NORM" \
  --lsm_path "$LSM" \
  --orography_path "$ORO" \
  --encdec_model_path "$AE" \
  --start_date "$TEST_START" \
  --end_date "$TEST_END" \
  --ensemble_size 10 \
  --num_inference_steps 20 \
  --return_seq_len 4 \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --input_seq_len 1 \
  --num_samples_per_month 10 \
  --save_as_latent \
  --sampler_type edm \
  --output "/home/limaocheng/ladcast/predictions/ladcast_terrain_ckpt${CKPT}_zero_infer_latent"
```

## 8. 评估全部预测

```bash
rm -rf \
  "metrics/ladcast_no_terrain_ckpt${CKPT}_latent" \
  "metrics/ladcast_terrain_ckpt${CKPT}_latent" \
  "metrics/ladcast_terrain_zero_ckpt${CKPT}_latent" \
  "metrics/ladcast_terrain_ckpt${CKPT}_zero_infer_latent"

"$PY" tools/eval_ladcast_latent_predictions.py \
  --pred_dir "predictions/ladcast_no_terrain_ckpt${CKPT}_latent" \
  --latent_path "$LATENT_PATH" \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --crop_init \
  --output "metrics/ladcast_no_terrain_ckpt${CKPT}_latent"

"$PY" tools/eval_ladcast_latent_predictions.py \
  --pred_dir "predictions/ladcast_terrain_ckpt${CKPT}_latent" \
  --latent_path "$LATENT_PATH" \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --crop_init \
  --output "metrics/ladcast_terrain_ckpt${CKPT}_latent"

"$PY" tools/eval_ladcast_latent_predictions.py \
  --pred_dir "predictions/ladcast_terrain_zero_ckpt${CKPT}_latent" \
  --latent_path "$LATENT_PATH" \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --crop_init \
  --output "metrics/ladcast_terrain_zero_ckpt${CKPT}_latent"

if [ -d "predictions/ladcast_terrain_ckpt${CKPT}_zero_infer_latent" ]; then
  "$PY" tools/eval_ladcast_latent_predictions.py \
    --pred_dir "predictions/ladcast_terrain_ckpt${CKPT}_zero_infer_latent" \
    --latent_path "$LATENT_PATH" \
    --total_lead_time_hour 240 \
    --step_size_hour 6 \
    --crop_init \
    --output "metrics/ladcast_terrain_ckpt${CKPT}_zero_infer_latent"
fi
```

## 9. 生成公平对比报告

```bash
"$PY" tools/compare_ladcast_baseline_metrics.py \
  --run_metrics "metrics/ladcast_terrain_ckpt${CKPT}_latent" \
  --baseline_metrics "metrics/ladcast_no_terrain_ckpt${CKPT}_latent" \
  --output_json "metrics/ladcast_terrain_ckpt${CKPT}_vs_no_terrain_ckpt${CKPT}_20180101_20180201.json"

"$PY" tools/compare_ladcast_baseline_metrics.py \
  --run_metrics "metrics/ladcast_terrain_ckpt${CKPT}_latent" \
  --baseline_metrics "metrics/ladcast_terrain_zero_ckpt${CKPT}_latent" \
  --output_json "metrics/ladcast_terrain_ckpt${CKPT}_vs_terrain_zero_ckpt${CKPT}_20180101_20180201.json"

if [ -d "metrics/ladcast_terrain_ckpt${CKPT}_zero_infer_latent" ]; then
  "$PY" tools/compare_ladcast_baseline_metrics.py \
    --run_metrics "metrics/ladcast_terrain_ckpt${CKPT}_latent" \
    --baseline_metrics "metrics/ladcast_terrain_ckpt${CKPT}_zero_infer_latent" \
    --output_json "metrics/ladcast_terrain_ckpt${CKPT}_vs_zero_infer_ckpt${CKPT}_20180101_20180201.json"
fi
```

## 10. 快速查看核心结果

```bash
"$PY" - <<'PY'
import json
import os
from pathlib import Path

ckpt = os.environ.get("CKPT", "60000")
paths = [
    Path(f"metrics/ladcast_terrain_ckpt{ckpt}_vs_no_terrain_ckpt{ckpt}_20180101_20180201.json"),
    Path(f"metrics/ladcast_terrain_ckpt{ckpt}_vs_terrain_zero_ckpt{ckpt}_20180101_20180201.json"),
    Path(f"metrics/ladcast_terrain_ckpt{ckpt}_vs_zero_infer_ckpt{ckpt}_20180101_20180201.json"),
]

for path in paths:
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    print(f"\n===== {path.name} =====")
    for metric in ["rmse_global", "rmse", "mae", "crps", "spread_skill_abs_error", "abs_bias"]:
        if metric not in report["paired_metric_reports"]:
            continue
        item = report["paired_metric_reports"][metric]
        print(
            f"{metric:24s} delta={item['delta_mean']:+.8f} "
            f"skill={item['relative_skill']*100:+.4f}% "
            f"CI=[{item['ci_low']:+.8f}, {item['ci_high']:+.8f}] "
            f"win={item['win_rate_by_initialization']:.3f}"
        )
PY
```

## 11. 判定口径

| 指标 | 更好方向 | 说明 |
|---|---:|---|
| `rmse_global` | 越低越好 | `sqrt(mean(MSE))`，和 `summary.json` 的 `rmse_mean` 同口径 |
| `rmse` | 越低越好 | `mean(sqrt(MSE))`，逐初始化/通道/lead 平均 |
| `mae` | 越低越好 | ensemble mean MAE |
| `crps` | 越低越好 | 概率预报质量，扩散/ensemble 核心指标 |
| `spread_skill_abs_error` | 越低越好 | `abs(spread/RMSE - 1)`，越接近 0 表示 spread-skill 越匹配 |
| `abs_bias` | 越低越好 | 平均偏差绝对值 |

主要论文结论建议优先看：

1. `terrain` vs `no_terrain`：说明相对原始 matched baseline 是否提升。
2. `terrain` vs `terrain_zero`：说明提升是否来自真实地形信息，而不是参数量增加。
3. `terrain` vs `zero_infer`：说明已训练模型推理时是否实际使用 terrain feature。
