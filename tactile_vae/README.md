# Hand-Wise Tactile VAE

Standalone VAE-like encoder/decoder training code for T-Rex tactile F6 windows.

## What It Trains

- Input sample: one hand, `[16, 5, 6]`.
- Raw source chunk: `[64, 5, 6]` at 30 Hz, subsampled as frames
  `[0, 4, 8, ..., 60]`.
- Encoder output: `mu, logvar`, each `[B, latent_dim]`.
- Decoder output: reconstructed `[B, 16, 5, 6]`.

The encoder is finger-aware internally but produces one hand-wise latent. It can
switch temporal aggregation with:

- `--temporal_pool attn`
- `--temporal_pool flatten_mlp`

Finger identity embeddings can be toggled with `--use_finger_embed 0/1`.

## Train

```bash
DATA_ROOT=/path/to/merged_midtrain_root \
OUTPUT_DIR=/path/to/outputs/tactile_vae \
bash tactile_vae/scripts/train_tactile_vae.sh
```

Useful overrides:

```bash
TEMPORAL_POOL=flatten_mlp USE_FINGER_EMBED=0 LATENT=128 EPOCHS=5 BATCH=128 \
DATA_ROOT=/path/to/merged OUTPUT_DIR=/tmp/tactile_vae \
bash tactile_vae/scripts/train_tactile_vae.sh
```

Smoke test:

```bash
python3 -m tactile_vae.train \
  --data_root /path/to/merged \
  --output_dir /tmp/tactile_vae_smoke \
  --smoke_test 1
```

## Checkpoint

Checkpoints contain the full model plus separately loadable encoder/decoder
weights:

```python
{
    "model_state": ...,
    "encoder_state": ...,
    "decoder_state": ...,
    "config": ...,
    "stats": ...,
    "step": ...,
    "epoch": ...,
}
```

Reuse:

```python
import torch
from tactile_vae.models import TactileVAE, TactileVAEConfig

state = torch.load("latest.pt", map_location="cpu", weights_only=False)
model = TactileVAE(TactileVAEConfig.from_dict(state["config"]))
model.load_state_dict(state["model_state"])
model.eval()

mu, logvar = model.encode(f6_normed)  # [B, latent_dim]
recon = model.decode(mu)              # deterministic latent by default
```

## Sanity Subset

For a tiny Hugging Face/LeRobot sanity pull without videos:

```bash
python3 tactile_vae/scripts/download_sanity_subset.py \
  --repo_id zekaiwang/trex_dataset \
  --episodes 0 \
  --max_gb 0.2
```

Then train with the LeRobot fallback loader if `lerobot` is installed:

```bash
python3 -m tactile_vae.train \
  --data_format lerobot \
  --data_root ~/.cache/trex_tactile_vae_sanity \
  --output_dir /tmp/tactile_vae_lerobot_smoke \
  --smoke_test 1
```

