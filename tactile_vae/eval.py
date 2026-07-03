"""Evaluate a trained hand-wise tactile VAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vae.data import TacF6Stats, build_train_val_datasets  # noqa: E402
from tactile_vae.models import TactileVAE, TactileVAEConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--exemplars", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TactileVAEConfig.from_dict(state["config"])
    stats = TacF6Stats.from_dict(state["stats"])
    model = TactileVAE(cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    _, val_ds, _ = build_train_val_datasets(
        data_root=args.data_root,
        source_window=cfg.source_window,
        input_window=cfg.input_window,
        subsample_stride=cfg.subsample_stride,
        stride=4,
        stats=stats,
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_ds.collate_fn,
    )

    sums: Dict[str, float] = {"recon": 0.0, "kl": 0.0, "total": 0.0, "mu_abs": 0.0, "mu_std": 0.0, "n": 0}
    mag, recon_ps = [], []
    ex_raw, ex_norm, ex_recon = [], [], []
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        f6 = batch["f6"].to(device)
        magnitude = batch["magnitude"].to(device)
        out = model(f6, magnitude, sample=False)
        n = f6.shape[0]
        sums["recon"] += float(out["recon_loss"].item()) * n
        sums["kl"] += float(out["kl_loss"].item()) * n
        sums["total"] += float(out["total_loss"].item()) * n
        sums["mu_abs"] += float(out["mu"].abs().mean().item()) * n
        sums["mu_std"] += float(out["mu"].std().item()) * n
        sums["n"] += n
        mag.append(batch["magnitude"].cpu().numpy())
        recon_ps.append(out["per_sample_recon"].cpu().numpy())
        if args.exemplars and len(ex_norm) < 4:
            take = min(4 - len(ex_norm), n)
            ex_raw.append(batch["raw_f6"][:take].cpu().numpy())
            ex_norm.append(batch["f6"][:take].cpu().numpy())
            ex_recon.append(out["recon"][:take].cpu().numpy())

    if sums["n"] == 0:
        raise RuntimeError("No validation samples evaluated.")
    summary = {
        "recon": sums["recon"] / sums["n"],
        "kl": sums["kl"] / sums["n"],
        "total": sums["total"] / sums["n"],
        "mu_abs": sums["mu_abs"] / sums["n"],
        "mu_std": sums["mu_std"] / sums["n"],
    }

    mags = np.concatenate(mag)
    recons = np.concatenate(recon_ps)
    edges = np.quantile(mags, [0.25, 0.5, 0.75])
    bins = np.digitize(mags, edges)
    for b in range(4):
        mask = bins == b
        summary[f"recon_mag_q{b + 1}"] = float(recons[mask].mean()) if mask.any() else float("nan")

    print(json.dumps(summary, indent=2))
    if args.exemplars:
        np.savez(
            args.exemplars,
            raw=np.concatenate(ex_raw, axis=0) if ex_raw else np.empty((0,)),
            norm=np.concatenate(ex_norm, axis=0) if ex_norm else np.empty((0,)),
            recon=np.concatenate(ex_recon, axis=0) if ex_recon else np.empty((0,)),
            summary=json.dumps(summary),
        )
        print(f"wrote exemplars -> {args.exemplars}")


if __name__ == "__main__":
    main()

