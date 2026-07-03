"""Train the hand-wise tactile VAE."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, random_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vae.data import (  # noqa: E402
    F6ChunkDataset,
    LeRobotF6ChunkDataset,
    TacF6Stats,
    build_train_val_datasets,
)
from tactile_vae.models import TactileVAE, TactileVAEConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--data_format", type=str, default="hdf5", choices=["hdf5", "lerobot"])
    p.add_argument("--lerobot_repo_id", type=str, default="zekaiwang/trex_dataset")
    p.add_argument("--source_window", type=int, default=64)
    p.add_argument("--input_window", type=int, default=16)
    p.add_argument("--subsample_stride", type=int, default=4)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--bottleneck_channels", type=int, default=256)
    p.add_argument("--bottleneck_T", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--n_strided_blocks", type=int, default=2)
    p.add_argument("--temporal_pool", type=str, default="attn", choices=["attn", "flatten_mlp"])
    p.add_argument("--use_finger_embed", type=int, default=1)
    p.add_argument("--use_time_embed", type=int, default=1)
    p.add_argument("--beta_kl", type=float, default=1e-3)
    p.add_argument("--use_magnitude_weight", type=int, default=1)
    p.add_argument("--weight_alpha", type=float, default=2.0)
    p.add_argument("--weight_tau", type=float, default=4.0)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--save_every_epoch", type=int, default=1)
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="trex_tactile_vae")
    p.add_argument("--smoke_test", type=int, default=0)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> TactileVAEConfig:
    return TactileVAEConfig(
        source_window=args.source_window,
        input_window=args.input_window,
        subsample_stride=args.subsample_stride,
        hidden_channels=args.hidden_channels,
        bottleneck_channels=args.bottleneck_channels,
        bottleneck_T=args.bottleneck_T,
        latent_dim=args.latent_dim,
        n_strided_blocks=args.n_strided_blocks,
        temporal_pool=args.temporal_pool,
        use_finger_embed=bool(args.use_finger_embed),
        use_time_embed=bool(args.use_time_embed),
        beta_kl=args.beta_kl,
        use_magnitude_weight=bool(args.use_magnitude_weight),
        weight_alpha=args.weight_alpha,
        weight_tau=args.weight_tau,
    )


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _save_checkpoint(
    accelerator: Accelerator,
    out_dir: str,
    model: TactileVAE,
    optimizer: torch.optim.Optimizer,
    stats: TacF6Stats,
    cfg: TactileVAEConfig,
    step: int,
    epoch: int,
):
    if not accelerator.is_main_process:
        return
    os.makedirs(out_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state = {
        "model_state": unwrapped.state_dict(),
        "encoder_state": unwrapped.encoder.state_dict(),
        "decoder_state": unwrapped.decoder.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "stats": stats.to_dict(),
        "step": step,
        "epoch": epoch,
    }
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(state, ckpt_path)
    torch.save(state, os.path.join(out_dir, "latest.pt"))
    accelerator.print(f"  saved checkpoint -> {ckpt_path}")


def _build_loaders(args: argparse.Namespace):
    if args.data_format == "hdf5":
        stats = TacF6Stats.from_data_root(args.data_root)
        train_ds, val_ds, stats = build_train_val_datasets(
            data_root=args.data_root,
            source_window=args.source_window,
            input_window=args.input_window,
            subsample_stride=args.subsample_stride,
            stride=args.stride,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stats=stats,
        )
    else:
        ds = LeRobotF6ChunkDataset(
            root=args.data_root,
            repo_id=args.lerobot_repo_id,
            source_window=args.source_window,
            input_window=args.input_window,
            subsample_stride=args.subsample_stride,
        )
        stats = ds.stats
        n_val = max(1, int(round(len(ds) * args.val_ratio)))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_ds.collate_fn = ds.collate_fn
        val_ds.collate_fn = ds.collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=getattr(train_ds, "collate_fn", F6ChunkDataset.collate_fn),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=getattr(val_ds, "collate_fn", F6ChunkDataset.collate_fn),
    )
    return train_ds, val_ds, train_loader, val_loader, stats


@torch.no_grad()
def _validate(
    model: TactileVAE,
    val_loader: DataLoader,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    sums = {"recon": 0.0, "kl": 0.0, "total": 0.0, "mu_abs": 0.0, "z_std": 0.0, "n": 0}
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(batch["f6"], batch["magnitude"], sample=False)
        n = batch["f6"].shape[0]
        sums["recon"] += float(out["recon_loss"].item()) * n
        sums["kl"] += float(out["kl_loss"].item()) * n
        sums["total"] += float(out["total_loss"].item()) * n
        sums["mu_abs"] += float(out["mu"].abs().mean().item()) * n
        sums["z_std"] += float(out["mu"].std().item()) * n
        sums["n"] += n
    model.train()
    if sums["n"] == 0:
        return {k: float("nan") for k in ("val_recon", "val_kl", "val_total", "val_mu_abs", "val_mu_std")}
    return {
        "val_recon": sums["recon"] / sums["n"],
        "val_kl": sums["kl"] / sums["n"],
        "val_total": sums["total"] / sums["n"],
        "val_mu_abs": sums["mu_abs"] / sums["n"],
        "val_mu_std": sums["z_std"] / sums["n"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    run_name = args.run_name or time.strftime("tactile_vae_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"[TactileVAE] run_dir = {run_dir}")
    accelerator.print(f"[TactileVAE] data_format = {args.data_format}")
    train_ds, val_ds, train_loader, val_loader, stats = _build_loaders(args)
    accelerator.print(
        f"[TactileVAE] train windows={len(train_ds)} val windows={len(val_ds)} "
        f"stats_source={stats.stats_source}")

    cfg = _build_config(args)
    model = TactileVAE(cfg)
    accelerator.print(f"[TactileVAE] config: {json.dumps(cfg.to_dict(), indent=2)}")
    accelerator.print(
        f"[TactileVAE] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-8,
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    use_wandb = bool(args.use_wandb) and accelerator.is_main_process
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=run_name, config={**vars(args), **cfg.to_dict()})
        except Exception as e:
            accelerator.print(f"[TactileVAE] wandb disabled: {e}")
            use_wandb = False

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            lr_now = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["f6"], batch["magnitude"], sample=True)
            accelerator.backward(out["total_loss"])
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step % args.log_every == 0 and accelerator.is_main_process:
                msg = (
                    f"[step {global_step:7d} | ep {epoch:2d}] "
                    f"loss={out['total_loss'].item():.4f} "
                    f"recon={out['recon_loss'].item():.4f} "
                    f"kl={out['kl_loss'].item():.4f} "
                    f"lr={lr_now:.2e} elapsed={time.time() - t0:.0f}s")
                accelerator.print(msg)
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": out["total_loss"].item(),
                        "train/recon": out["recon_loss"].item(),
                        "train/kl": out["kl_loss"].item(),
                        "lr": lr_now,
                        "epoch": epoch,
                    }, step=global_step)

            if args.val_every > 0 and global_step > 0 and global_step % args.val_every == 0:
                vals = _validate(accelerator.unwrap_model(model), val_loader, max_batches=50)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [val @ step {global_step}] " +
                        " ".join(f"{k}={v:.4f}" for k, v in vals.items()))
                    if use_wandb:
                        import wandb
                        wandb.log({f"val/{k[4:]}": v for k, v in vals.items()}, step=global_step)

            global_step += 1
            if args.smoke_test and global_step >= 5:
                break
        if args.smoke_test:
            break
        if (epoch + 1) % args.save_every_epoch == 0:
            _save_checkpoint(accelerator, run_dir, model, optimizer, stats, cfg, global_step, epoch)

    _save_checkpoint(
        accelerator,
        run_dir,
        model,
        optimizer,
        stats,
        cfg,
        global_step,
        epoch=args.epochs - 1 if not args.smoke_test else 0,
    )
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()
    accelerator.print(f"[TactileVAE] Done. Total wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

