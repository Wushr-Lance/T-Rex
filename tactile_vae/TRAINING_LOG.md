# Tactile VAE Training Log

Date: 2026-07-05 UTC

Repository: `/home/shenrui/egoscale2/T-Rex`

## Current Status

Formal training is running in tmux session `tactile_vae_train`.

- Run name: `tactile_vae_trex_parquet_2gpu_20260705_0718`
- W&B project: `berkeley_bair/trex_tactile_vae`
- W&B run: `https://wandb.ai/berkeley_bair/trex_tactile_vae/runs/5b3utpyd`
- Output directory: `/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/tactile_vae_trex_parquet_2gpu_20260705_0718`
- Training log: `/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/logs/tactile_vae_trex_parquet_2gpu_20260705_0718_20260705_071745.log`
- Latest checkpoint: `/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/tactile_vae_trex_parquet_2gpu_20260705_0718/latest.pt`
- First epoch checkpoint: `/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/tactile_vae_trex_parquet_2gpu_20260705_0718/checkpoint_epoch000.pt`

Monitor commands:

```bash
tmux attach -t tactile_vae_train
tail -f /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/logs/tactile_vae_trex_parquet_2gpu_20260705_0718_20260705_071745.log
```

## Code Changes Made

- Added YAML config support to `tactile_vae/train.py`.
  - New public flag: `--config`.
  - YAML defaults are loaded first; command-line flags override YAML values.
  - Supported sections: `paths`, `data`, `model`, `loss`, `train`, `logging`.
- Added W&B entity support to `tactile_vae/train.py`.
  - New public flag: `--wandb_entity`.
  - `wandb.init(...)` now receives `entity=args.wandb_entity`.
- Added benchmark/short-run support to `tactile_vae/train.py`.
  - New public flag: `--max_steps`.
  - Training logs now include `samples/s` and cumulative `examples`.
  - `--max_steps` exits cleanly and saves the checkpoint with the actual epoch reached.
- Added config file:
  - `tactile_vae/config/tactile_vae_trex_parquet.yaml`
- Updated launcher:
  - `tactile_vae/scripts/train_tactile_vae.sh`
  - Accepts `CONFIG`, `DATA_FORMAT`, `WANDB_ENTITY`, `WANDB_PROJECT`, `MAX_STEPS`, `WANDB_MODE`, and no-video parquet defaults.
  - Defaults output to `/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs`.
- Added `PyYAML==6.0.2` to `pyproject.toml`.
- Added focused tests:
  - `tests/test_tactile_vae_train_config.py`

Verification commands run:

```bash
conda run -n trex python -m unittest tests/test_tactile_vae_train_config.py
conda run -n trex python -m tactile_vae.train --help | rg -- '--config|--wandb_entity|--max_steps'
bash -n tactile_vae/scripts/train_tactile_vae.sh
```

## Conda Environment

The previous `trex` env was deleted and recreated.

Commands used:

```bash
conda env remove -n trex -y
conda create -n trex python=3.10 -y
conda run -n trex pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n trex pip install -e .
```

Verified package versions:

```text
python 3.10.20
torch 2.6.0+cu124
cuda_available True
cuda_count 8
h5py 3.16.0
numpy 2.2.6
pandas 2.3.3
pyarrow 20.0.0
accelerate 1.8.1
wandb 0.18.3
huggingface_hub 0.36.2
yaml 6.0.2
```

W&B auth verification:

```bash
conda run -n trex wandb login --verify
```

Observed:

```text
wandb: Currently logged in as: wushr-lance (berkeley_bair).
```

## Dataset Download

Downloaded the T-Rex Hugging Face dataset without videos.

Command used:

```bash
conda run -n trex python -c "from huggingface_hub import snapshot_download; root=snapshot_download(repo_id='zekaiwang/trex_dataset', repo_type='dataset', local_dir='/data/d3/shenrui/trex_dataset_no_videos', allow_patterns=['meta/**','data/**']); print(root)"
```

Observed local root:

```text
/data/local/shenrui/trex_dataset_no_videos
```

This is the same mounted storage as the requested path:

```text
/data/d3/shenrui/trex_dataset_no_videos
```

Dataset verification:

```bash
du -sh /data/d3/shenrui/trex_dataset_no_videos
find /data/d3/shenrui/trex_dataset_no_videos/data -name '*.parquet' | wc -l
test ! -d /data/d3/shenrui/trex_dataset_no_videos/videos
test -f /data/d3/shenrui/trex_dataset_no_videos/meta/info.json
test -f /data/d3/shenrui/trex_dataset_no_videos/meta/stats.json
```

Observed:

```text
3.1G /data/d3/shenrui/trex_dataset_no_videos
48 parquet files
no videos directory
meta/info.json exists
meta/stats.json exists
```

Parquet schema check:

```bash
conda run -n trex python -c "import pandas as pd; p='/data/d3/shenrui/trex_dataset_no_videos/data/chunk-000/file-000.parquet'; df=pd.read_parquet(p); print('shape', df.shape); print('has_tactile_force', 'observation.tactile_force' in df.columns); print('first_tactile_shape', df.iloc[0]['observation.tactile_force'].shape); print('episodes', df['episode_index'].nunique())"
```

Observed:

```text
shape (122652, 8)
has_tactile_force True
first_tactile_shape (60,)
episodes 144
```

## Smoke Test

Command used:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n trex python -m tactile_vae.train \
  --config tactile_vae/config/tactile_vae_trex_parquet.yaml \
  --run_name smoke_real_parquet \
  --output_dir /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
  --use_wandb 0 \
  --max_steps 1 \
  --num_workers 0 \
  --batch_size 8 \
  --hidden_channels 32 \
  --bottleneck_channels 64 \
  --latent_dim 64 \
  --bottleneck_T 4 \
  --mixed_precision no \
  --log_every 1
```

Observed:

```text
train windows=2517556 val windows=51220 stats_source=lerobot_stats_json
[step       0 | ep  0] loss=0.7493 recon=0.7493 kl=0.0180
saved checkpoint -> /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/smoke_real_parquet/checkpoint_epoch000.pt
```

## Benchmarks

First attempt:

```bash
CUDA_VISIBLE_DEVICES=0 USE_WANDB=0 MAX_STEPS=100 RUN_NAME=bench_1gpu_100 OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs bash tactile_vae/scripts/train_tactile_vae.sh
```

This failed because the base shell did not have `accelerate` on `PATH`. The corrected benchmark commands were run through `conda run -n trex`.

1-GPU benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 USE_WANDB=0 MAX_STEPS=100 RUN_NAME=bench_1gpu_100 OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

Observed steady log point:

```text
[step      50 | ep  0] loss=0.5009 recon=0.5008 kl=0.0526 lr=3.06e-05 samples/s=31118.4 examples=13056
```

2-GPU benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1 USE_WANDB=0 MAX_STEPS=100 RUN_NAME=bench_2gpu_100 OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

Observed steady log point:

```text
[step      50 | ep  0] loss=0.4559 recon=0.4558 kl=0.0550 lr=3.06e-05 samples/s=55066.5 examples=26112
```

Decision:

- 2 GPUs were selected for formal training.
- Reason: `55066.5 / 31118.4 = 1.77x`, which is above the planned `1.3x` threshold.

## W&B Online Probe

Before launching the long run, a 1-step W&B-online probe was run.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 USE_WANDB=1 WANDB_MODE=online WANDB_ENTITY=berkeley_bair WANDB_PROJECT=trex_tactile_vae MAX_STEPS=1 RUN_NAME=wandb_online_probe OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

Observed:

```text
wandb: View project at https://wandb.ai/berkeley_bair/trex_tactile_vae
wandb: View run at https://wandb.ai/berkeley_bair/trex_tactile_vae/runs/n0nagoma
[step       0 | ep  0] loss=0.7304 recon=0.7304 kl=0.0108
wandb: Run history included train/loss
```

## Formal Training Launch

Two `nohup`-style attempts exited immediately with empty outer logs in this execution environment:

```text
/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/logs/tactile_vae_trex_parquet_2gpu_20260705_0716_outer.log
/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/logs/tactile_vae_trex_parquet_2gpu_20260705_0717_outer.log
```

The persistent launch was then done with tmux:

```bash
tmux new-session -d -s tactile_vae_train "cd /home/shenrui/egoscale2/T-Rex; CUDA_VISIBLE_DEVICES=0,1 USE_WANDB=1 WANDB_MODE=online WANDB_ENTITY=berkeley_bair WANDB_PROJECT=trex_tactile_vae RUN_NAME=tactile_vae_trex_parquet_2gpu_20260705_0718 OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache conda run --no-capture-output -n trex bash tactile_vae/scripts/train_tactile_vae.sh"
```

Observed process roots:

```text
tmux session: tactile_vae_train
conda run pid: 918864
train shell pid: 919082
accelerate pid: 919092
worker pids include: 919865, 919866
```

Observed W&B sync:

```text
wandb: View project at https://wandb.ai/berkeley_bair/trex_tactile_vae
wandb: View run at https://wandb.ai/berkeley_bair/trex_tactile_vae/runs/5b3utpyd
[step       0 | ep  0] loss=0.7434 recon=0.7433 kl=0.0109 lr=6.00e-07 samples/s=669.8 examples=512
[step      50 | ep  0] loss=0.4559 recon=0.4558 kl=0.0550 lr=3.06e-05 samples/s=60104.8 examples=26112
```

Observed validation and checkpoint:

```text
[val @ step 4000] val_recon=0.0057 val_kl=1.8598 val_total=0.0076 val_mu_abs=0.4167 val_mu_std=0.5370
saved checkpoint -> /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/tactile_vae_trex_parquet_2gpu_20260705_0718/checkpoint_epoch000.pt
```

## Notes

- Videos were not downloaded.
- The trainer is reading `observation.tactile_force` from local parquet files via `ParquetF6ChunkDataset`.
- Normalization stats came from the downloaded LeRobot `meta/stats.json` (`stats_source=lerobot_stats_json`).
- `TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache` was set for training to avoid using the default NFS home cache.
