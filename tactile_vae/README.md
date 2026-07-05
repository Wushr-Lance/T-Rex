# Hand-Wise Tactile VAE

This folder is a standalone training codebase for a **hand-wise VAE-like
tactile force encoder/decoder**. It is designed for T-Rex tactile F6 data and
can be trained separately from the main VLA model.

The model learns:

```text
one hand tactile force window [16, 5, 6]
    -> encoder
mu, logvar [latent_dim]
    -> sample or use mu
z [latent_dim]
    -> decoder
reconstruction [16, 5, 6]
```

The final latent is **one vector per hand**, but the encoder and decoder are
finger-aware internally.

## 1. What The Model Consumes

T-Rex tactile force is stored per frame as:

```text
[10, 6]
```

where:

```text
10 = 2 hands x 5 fingers
6  = Fx, Fy, Fz, Mx, My, Mz
```

Training treats left and right hands as independent samples:

```text
raw bimanual chunk: [64, 10, 6]
select one hand:    [64, 5, 6]
subsample stride 4: [16, 5, 6]
```

At 30 Hz, 64 raw frames cover about 2.13 seconds. The default model samples
frames:

```text
[0, 4, 8, ..., 60]
```

so the model input has 16 frames.

Normalization follows T-Rex q01/q99 min-max normalization to `[-1, 1]`. For
hand-wise samples, left hand uses stats entries `0:30`; right hand uses `30:60`.

## 2. Folder Layout

```text
tactile_vae/
├── data/
│   ├── dataset.py       HDF5, LeRobot, and local parquet tactile loaders
│   └── stats.py         T-Rex q01/q99 F6 normalization
├── models/
│   ├── encoder.py       finger-aware hand-wise VAE encoder
│   ├── decoder.py       finger-aware hand-wise decoder
│   └── tactile_vae.py   VAE wrapper, config, losses
├── scripts/
│   ├── download_sanity_subset.py
│   └── train_tactile_vae.sh
├── train.py
├── eval.py
└── README.md
```

## 3. Installation

From the repository root:

```bash
cd /path/to/T-Rex
```

Create the environment using the same dependency style as T-Rex:

```bash
conda create -n trex python=3.10 -y
conda activate trex

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -e .
```

Optional but recommended for the LeRobot path:

```bash
pip install -e /path/to/lerobot
```

Confirm the core packages:

```bash
python - <<'PY'
import torch, h5py, numpy, pandas, pyarrow, accelerate, wandb
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("ok")
PY
```

If you only want to reproduce the local parquet sanity run, you need:

```text
torch, numpy, pandas, pyarrow, huggingface_hub, accelerate, wandb
```

## 4. Quick Sanity Dataset Download

The full T-Rex dataset is large, so start by downloading a small no-video subset.
This pulls metadata plus trajectory parquet files containing tactile force, not
camera/tactile videos.

```bash
python tactile_vae/scripts/download_sanity_subset.py \
  --repo_id zekaiwang/trex_dataset \
  --cache_dir outputs/tactile_vae_sanity_data \
  --episodes 0 \
  --max_gb 0.2
```

Expected result:

```text
outputs/tactile_vae_sanity_data/
├── data/**/*.parquet
└── meta/
```

Check that tactile force exists:

```bash
python - <<'PY'
import pandas as pd
p = "outputs/tactile_vae_sanity_data/data/chunk-000/file-000.parquet"
df = pd.read_parquet(p)
print(df.shape)
print("observation.tactile_force" in df.columns)
print(df.iloc[0]["observation.tactile_force"].shape)
PY
```

You should see `observation.tactile_force` with shape `(60,)` per frame.

## 5. Smoke Train On The Sanity Subset

Use the direct local parquet loader:

```bash
WANDB_MODE=offline \
WANDB_DIR=outputs/wandb \
WANDB_CACHE_DIR=outputs/wandb_cache \
PYTHONPATH=$PWD \
python -m tactile_vae.train \
  --data_format parquet \
  --data_root outputs/tactile_vae_sanity_data \
  --output_dir outputs/tactile_vae_runs \
  --run_name sanity_parquet_smoke \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 0 \
  --latent_dim 64 \
  --hidden_channels 32 \
  --bottleneck_channels 64 \
  --bottleneck_T 4 \
  --temporal_pool attn \
  --use_finger_embed 1 \
  --val_ratio 0.2 \
  --stride 64 \
  --log_every 1 \
  --val_every 2 \
  --use_wandb 1 \
  --mixed_precision no \
  --smoke_test 1
```

This is intentionally tiny and CPU-friendly. It verifies:

- dataset loading
- `[64, 10, 6] -> [16, 5, 6]` chunking
- q01/q99 normalization
- encoder/decoder shape flow
- VAE losses
- checkpoint writing
- W&B offline logging

Expected checkpoint:

```text
outputs/tactile_vae_runs/sanity_parquet_smoke/latest.pt
```

If W&B is offline, sync later with:

```bash
wandb sync outputs/wandb/wandb/offline-run-*
```

## 6. Full Training On Merged T-Rex HDF5 Data

The full training path follows the same layout as the existing tactile VQ-VAE:

```text
DATA_ROOT/
  split_or_source_name/
    pretrain_manifest.json
  episode_dir/
    pretrain.hdf5
      tactile_f6: [N, 10, 6]
```

The manifest must include tactile q01/q99 statistics under:

```text
statistics.tactile_f6.q01
statistics.tactile_f6.q99
```

Run with the shell launcher:

```bash
DATA_ROOT=/path/to/merged_midtrain_root \
OUTPUT_DIR=outputs/tactile_vae \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_WANDB=1 \
WANDB_API_KEY=... \
bash tactile_vae/scripts/train_tactile_vae.sh
```

Useful overrides:

```bash
TEMPORAL_POOL=flatten_mlp \
USE_FINGER_EMBED=0 \
LATENT=128 \
BATCH=128 \
EPOCHS=10 \
LR=3e-4 \
DATA_ROOT=/path/to/merged_midtrain_root \
OUTPUT_DIR=outputs/tactile_vae \
bash tactile_vae/scripts/train_tactile_vae.sh
```

The launcher uses:

```text
SOURCE_WINDOW=64
INPUT_WINDOW=16
SUBSAMPLE_STRIDE=4
STRIDE=4
TEMPORAL_POOL=attn
USE_FINGER_EMBED=1
LATENT=256
BATCH=256
EPOCHS=30
LR=3e-4
```

For full training, `STRIDE=4` gives many overlapping 64-frame chunks. Increase
it, for example to `STRIDE=16` or `STRIDE=64`, if you want faster but less dense
training.

## 7. Direct Python Training Command

Equivalent HDF5 training without the shell script:

```bash
PYTHONPATH=$PWD accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m tactile_vae.train \
  --data_format hdf5 \
  --data_root /path/to/merged_midtrain_root \
  --output_dir outputs/tactile_vae \
  --run_name tactile_vae_full \
  --source_window 64 \
  --input_window 16 \
  --subsample_stride 4 \
  --stride 4 \
  --temporal_pool attn \
  --use_finger_embed 1 \
  --latent_dim 256 \
  --hidden_channels 128 \
  --bottleneck_channels 256 \
  --bottleneck_T 4 \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-4 \
  --num_workers 4 \
  --val_every 2000 \
  --use_wandb 1
```

For CPU-only debugging, use:

```bash
python -m tactile_vae.train \
  --data_format parquet \
  --data_root outputs/tactile_vae_sanity_data \
  --output_dir /tmp/tactile_vae_debug \
  --mixed_precision no \
  --num_workers 0 \
  --batch_size 4 \
  --smoke_test 1
```

## 8. Evaluation

Evaluate a checkpoint on HDF5 data:

```bash
python -m tactile_vae.eval \
  --data_format hdf5 \
  --checkpoint outputs/tactile_vae/<run_name>/latest.pt \
  --data_root /path/to/merged_midtrain_root \
  --batch_size 256 \
  --max_batches 100 \
  --exemplars outputs/tactile_vae/<run_name>/eval_exemplars.npz
```

Evaluate a sanity parquet checkpoint:

```bash
python -m tactile_vae.eval \
  --data_format parquet \
  --checkpoint outputs/tactile_vae_runs/sanity_parquet_smoke/latest.pt \
  --data_root outputs/tactile_vae_sanity_data \
  --batch_size 16 \
  --max_batches 10 \
  --exemplars outputs/tactile_vae_runs/sanity_parquet_smoke/eval_exemplars.npz
```

The eval summary reports:

```text
recon          reconstruction MSE
kl             KL to N(0, I)
total          recon + beta_kl * KL
mu_abs         average absolute latent mean
mu_std         latent mean standard deviation
recon_mag_q*   reconstruction MSE by raw-force magnitude quartile
```

## 9. Checkpoint Format

Each checkpoint contains:

```python
{
    "model_state":   full_model_state_dict,
    "encoder_state": encoder_state_dict,
    "decoder_state": decoder_state_dict,
    "optim_state":   optimizer_state_dict,
    "config":        config_dict,
    "stats":         normalization_stats_dict,
    "step":          global_step,
    "epoch":         epoch,
}
```

Load the full model:

```python
import torch
from tactile_vae.models import TactileVAE, TactileVAEConfig

ckpt = torch.load("latest.pt", map_location="cpu", weights_only=False)
cfg = TactileVAEConfig.from_dict(ckpt["config"])
model = TactileVAE(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

Use the encoder as a deterministic hand-wise tactile representation:

```python
with torch.no_grad():
    mu, logvar = model.encode(f6_normed)  # f6_normed: [B, 16, 5, 6]
    z = mu                                # recommended deterministic export
```

Decode:

```python
with torch.no_grad():
    recon = model.decode(z)               # [B, 16, 5, 6]
```

If another codebase only needs the encoder:

```python
model.encoder.load_state_dict(ckpt["encoder_state"])
```

Remember to apply the saved T-Rex q01/q99 stats before encoding.

## 10. Architecture Summary

Encoder:

```text
[B, 16, 5, 6]
  -> treat each finger as batch
[B*5, 6, 16]
  -> shared temporal Conv1d stack
[B*5, bottleneck, 4]
  -> temporal_pool={attn, flatten_mlp}
[B*5, bottleneck]
  -> restore fingers
[B, 5, bottleneck]
  -> finger attention pooling
[B, bottleneck]
  -> mu/logvar heads
[B, latent_dim], [B, latent_dim]
```

Decoder:

```text
z [B, latent_dim]
  -> learned latent_to_tokens
[B, 5, bottleneck, 4]
  -> add optional finger/time embeddings
  -> shared ConvTranspose1d decoder per finger
[B*5, 6, 16]
  -> reshape
[B, 16, 5, 6]
```

Training loss:

```text
total_loss = reconstruction_loss + beta_kl * KL(q(z|x) || N(0, I))
```

Default:

```text
beta_kl = 1e-3
```

## 11. Important Configs

Data:

```text
--source_window 64       raw frames per chunk
--input_window 16        frames after subsampling
--subsample_stride 4     64 -> 16
--stride 4               chunk-start stride inside an episode
```

Model:

```text
--latent_dim 256
--hidden_channels 128
--bottleneck_channels 256
--bottleneck_T 4
--temporal_pool attn | flatten_mlp
--use_finger_embed 0 | 1
--use_time_embed 0 | 1
```

Loss:

```text
--beta_kl 1e-3
--use_magnitude_weight 0 | 1
--weight_alpha 2.0
--weight_tau 4.0
```

Logging:

```text
--use_wandb 1
--wandb_project trex_tactile_vae
```

## 12. Troubleshooting

**`No pretrain_manifest.json`**

You are using `--data_format hdf5` on a LeRobot/parquet root. Use:

```bash
--data_format parquet
```

for the sanity subset downloaded by `download_sanity_subset.py`.

**LeRobot tries to contact Hugging Face after local download**

Use the direct parquet loader:

```bash
--data_format parquet
```

This avoids LeRobot metadata compatibility issues and reads local parquet files
directly.

**W&B cannot open sockets in a restricted environment**

Run training outside the sandbox, or use offline mode with W&B directories inside
the workspace:

```bash
WANDB_MODE=offline \
WANDB_DIR=outputs/wandb \
WANDB_CACHE_DIR=outputs/wandb_cache \
python -m tactile_vae.train ...
```

**CUDA is not visible**

Check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

If this prints `False`, the code still runs on CPU for smoke tests, but full
training should be run in a GPU-visible environment.

**Loss is finite but not improving in a smoke test**

That is normal. `--smoke_test 1` only runs five optimizer steps. Use it only to
validate wiring. Real training needs more epochs and a larger dataset.

