# Tactile VAE

`tactile_vae/` is a standalone trainer for a hand-wise VAE over T-Rex tactile
force windows. It is separate from the main VLA training code and learns a
continuous latent representation for one hand of tactile F6 force/torque data.

The default training path uses the public T-Rex Hugging Face dataset in
LeRobot parquet format, downloading only `meta/**` and `data/**`. Camera and
tactile videos are not needed for this model.

For an exact record of the first full run on this machine, including commands,
logs, benchmark results, and W&B links, see [`TRAINING_LOG.md`](TRAINING_LOG.md).

## Model

T-Rex tactile force is stored per frame as `[10, 6]`:

```text
10 = 2 hands x 5 fingers
6  = Fx, Fy, Fz, Mx, My, Mz
```

Training treats the two hands as independent samples:

```text
raw bimanual chunk: [64, 10, 6]
select one hand:    [64, 5, 6]
subsample stride 4: [16, 5, 6]
```

The VAE learns:

```text
[B, 16, 5, 6]
  -> encoder
mu, logvar [B, latent_dim]
  -> reparameterize or use mu
z [B, latent_dim]
  -> decoder
reconstruction [B, 16, 5, 6]
```

The final latent is one vector per hand. The encoder and decoder are still
finger-aware internally. Normalization follows T-Rex q01/q99 min-max scaling to
`[-1, 1]`; left-hand samples use stats entries `0:30`, right-hand samples use
entries `30:60`.

Training loss:

```text
total_loss = reconstruction_loss + beta_kl * KL(q(z|x) || N(0, I))
```

Default `beta_kl` is `1e-3`.

## Layout

```text
tactile_vae/
├── config/
│   └── tactile_vae_trex_parquet.yaml   default no-video T-Rex training config
├── data/
│   ├── dataset.py                      HDF5, LeRobot, and local parquet loaders
│   └── stats.py                        T-Rex q01/q99 F6 normalization
├── models/
│   ├── encoder.py                      finger-aware temporal encoder
│   ├── decoder.py                      finger-aware temporal decoder
│   └── tactile_vae.py                  VAE wrapper, config, and losses
├── scripts/
│   ├── download_sanity_subset.py        tiny per-episode no-video download helper
│   └── train_tactile_vae.sh            accelerate launcher
├── train.py                            main training entrypoint
├── eval.py                             checkpoint evaluation
├── TRAINING_LOG.md                     detailed local run record
└── README.md
```

## Environment

From the repository root:

```bash
cd /path/to/T-Rex
conda create -n trex python=3.10 -y
conda activate trex

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -e .
```

Verify the environment:

```bash
python - <<'PY'
import torch, h5py, numpy, pandas, pyarrow, accelerate, wandb, yaml
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("ok")
PY
```

The default launcher assumes the env is active. If it is not active, wrap
commands with `conda run -n trex ...`.

## Download The No-Video Dataset

Download only metadata and trajectory parquet files:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zekaiwang/trex_dataset",
    repo_type="dataset",
    local_dir="/data/d3/shenrui/trex_dataset_no_videos",
    allow_patterns=["meta/**", "data/**"],
)
PY
```

Expected local structure:

```text
/data/d3/shenrui/trex_dataset_no_videos/
├── data/**/*.parquet
└── meta/
```

Check that videos were not downloaded and tactile force exists:

```bash
test ! -d /data/d3/shenrui/trex_dataset_no_videos/videos
test -f /data/d3/shenrui/trex_dataset_no_videos/meta/info.json
test -f /data/d3/shenrui/trex_dataset_no_videos/meta/stats.json
find /data/d3/shenrui/trex_dataset_no_videos/data -name '*.parquet' | wc -l

python - <<'PY'
import pandas as pd

p = "/data/d3/shenrui/trex_dataset_no_videos/data/chunk-000/file-000.parquet"
df = pd.read_parquet(p)
print(df.shape)
print("observation.tactile_force" in df.columns)
print(df.iloc[0]["observation.tactile_force"].shape)
PY
```

`observation.tactile_force` should be present with shape `(60,)` per frame.

## Configuration

The default full-run config is:

```text
tactile_vae/config/tactile_vae_trex_parquet.yaml
```

It contains these sections:

```text
paths      data root, output directory, run name
data       parquet/HDF5 mode, window sizes, stride, workers, validation split
model      latent/model dimensions and pooling options
loss       KL weight and magnitude weighting
train      epochs, batch size, LR, precision, seed, max_steps, sample_latent
logging    W&B project/entity and logging cadence
```

Command-line flags override YAML values. For example:

```bash
python -m tactile_vae.train \
  --config tactile_vae/config/tactile_vae_trex_parquet.yaml \
  --batch_size 128 \
  --max_steps 100 \
  --use_wandb 0
```

Useful flags:

```text
--config PATH
--data_root PATH
--data_format parquet|hdf5|lerobot
--output_dir PATH
--run_name NAME
--max_steps N
--use_wandb 0|1
--wandb_project trex_tactile_vae
--wandb_entity berkeley_bair
```

The shell launcher intentionally passes only runtime overrides such as
`RUN_NAME`, `OUTPUT_DIR`, and W&B settings. Training hyperparameters such as
`batch_size`, `lr`, `epochs`, `max_steps`, `latent_dim`, and window/stride
settings should be edited in the YAML config.

`sample_latent: 1` uses VAE reparameterization during training:

```text
z = mu + eps * std
```

Set `sample_latent: 0` to train the decoder on deterministic `z = mu`. Validation
already uses deterministic `z = mu`.

## Smoke Test

Run a small non-W&B check on the downloaded parquet data:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m tactile_vae.train \
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
  --mixed_precision no \
  --log_every 1
```

Expected output includes:

```text
data_format = parquet
train windows=... val windows=... stats_source=lerobot_stats_json
[step       0 | ep  0] loss=...
saved checkpoint -> .../checkpoint_epoch000.pt
```

## Full Training

Use the launcher for normal training:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
USE_WANDB=1 \
WANDB_MODE=online \
WANDB_ENTITY=berkeley_bair \
WANDB_PROJECT=trex_tactile_vae \
RUN_NAME=tactile_vae_trex_parquet_2gpu \
OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache \
bash tactile_vae/scripts/train_tactile_vae.sh
```

If the conda env is not active:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
USE_WANDB=1 \
WANDB_MODE=online \
WANDB_ENTITY=berkeley_bair \
WANDB_PROJECT=trex_tactile_vae \
RUN_NAME=tactile_vae_trex_parquet_2gpu \
OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache \
conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

The launcher mirrors stdout/stderr to:

```text
<OUTPUT_DIR>/logs/<RUN_NAME>_<timestamp>.log
```

For a persistent detached run:

```bash
tmux new-session -d -s tactile_vae_train \
  "cd /home/shenrui/egoscale2/T-Rex; \
   CUDA_VISIBLE_DEVICES=0,1 \
   USE_WANDB=1 \
   WANDB_MODE=online \
   WANDB_ENTITY=berkeley_bair \
   WANDB_PROJECT=trex_tactile_vae \
   RUN_NAME=tactile_vae_trex_parquet_2gpu \
   OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
   TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache \
   conda run --no-capture-output -n trex bash tactile_vae/scripts/train_tactile_vae.sh"
```

Monitor:

```bash
tmux attach -t tactile_vae_train
tail -f /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/logs/<RUN_NAME>_<timestamp>.log
```

## Benchmark Before Choosing GPU Count

This model is small, so benchmark before using many GPUs:

```bash
cp tactile_vae/config/tactile_vae_trex_parquet.yaml /tmp/tactile_vae_bench.yaml
python - <<'PY'
from pathlib import Path

p = Path("/tmp/tactile_vae_bench.yaml")
s = p.read_text()
s = s.replace("max_steps: 0", "max_steps: 100")
p.write_text(s)
PY

CUDA_VISIBLE_DEVICES=0 \
USE_WANDB=0 \
CONFIG=/tmp/tactile_vae_bench.yaml \
RUN_NAME=bench_1gpu_100 \
OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh

CUDA_VISIBLE_DEVICES=0,1 \
USE_WANDB=0 \
CONFIG=/tmp/tactile_vae_bench.yaml \
RUN_NAME=bench_2gpu_100 \
OUTPUT_DIR=/home/shenrui/egoscale2/T-Rex/tactile_vae/outputs \
conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

Use 1 GPU if it is fast enough or if 2 GPUs improve throughput by less than
about `1.3x`. In the recorded local run, 2 GPUs were used because they achieved
about `1.77x` the 1-GPU throughput.

## Evaluation

Evaluate a parquet-trained checkpoint:

```bash
python -m tactile_vae.eval \
  --data_format parquet \
  --checkpoint /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/<run_name>/latest.pt \
  --data_root /data/d3/shenrui/trex_dataset_no_videos \
  --batch_size 256 \
  --max_batches 100 \
  --exemplars /home/shenrui/egoscale2/T-Rex/tactile_vae/outputs/<run_name>/eval_exemplars.npz
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

## Checkpoints

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

Use the encoder deterministically:

```python
with torch.no_grad():
    mu, logvar = model.encode(f6_normed)  # f6_normed: [B, 16, 5, 6]
    z = mu
```

`f6_normed` must use the saved q01/q99 stats from the checkpoint.

## Troubleshooting

**`No pretrain_manifest.json`**

You are using `--data_format hdf5` on a parquet root. Use:

```bash
--data_format parquet
```

or use `tactile_vae/config/tactile_vae_trex_parquet.yaml`.

**`accelerate: command not found`**

The conda env is not active. Either activate it:

```bash
conda activate trex
```

or run the launcher through:

```bash
conda run -n trex bash tactile_vae/scripts/train_tactile_vae.sh
```

**W&B logs offline**

Set:

```bash
USE_WANDB=1
WANDB_MODE=online
WANDB_ENTITY=berkeley_bair
```

Confirm auth:

```bash
wandb login --verify
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

**Loss is finite but not improving in a smoke test**

That is expected. Smoke tests are only wiring checks. Use the full dataset and
more steps for real training.
