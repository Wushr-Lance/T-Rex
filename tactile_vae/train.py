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
import yaml
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
    ParquetF6ChunkDataset,
    TacF6Stats,
    build_train_val_datasets,
)
from tactile_vae.models import TactileVAE, TactileVAEConfig  # noqa: E402


_DEFAULTS = {
    "data_format": "hdf5",
    "lerobot_repo_id": "zekaiwang/trex_dataset",
    "source_window": 64,
    "input_window": 16,
    "subsample_stride": 4,
    "stride": 4,
    "val_ratio": 0.02,
    "num_workers": 4,
    "hidden_channels": 128,
    "bottleneck_channels": 256,
    "bottleneck_T": 4,
    "latent_dim": 256,
    "n_strided_blocks": 2,
    "temporal_pool": "attn",
    "use_finger_embed": 1,
    "use_time_embed": 1,
    "beta_kl": 1e-3,
    "use_magnitude_weight": 1,
    "weight_alpha": 2.0,
    "weight_tau": 4.0,
    "run_name": None,
    "epochs": 30,
    "batch_size": 256,
    "lr": 3e-4,
    "min_lr_ratio": 0.05,
    "warmup_steps": 500,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "mixed_precision": "bf16",
    "seed": 42,
    "max_steps": 0,
    "log_every": 50,
    "val_every": 2000,
    "save_every_epoch": 1,
    "use_wandb": 0,
    "wandb_project": "trex_tactile_vae",
    "wandb_entity": None,
    "smoke_test": 0,
}

_CONFIG_KEYS = {
    "paths": ("data_root", "output_dir", "run_name"),
    "data": (
        "data_format",
        "lerobot_repo_id",
        "source_window",
        "input_window",
        "subsample_stride",
        "stride",
        "val_ratio",
        "num_workers",
    ),
    "model": (
        "hidden_channels",
        "bottleneck_channels",
        "bottleneck_T",
        "latent_dim",
        "n_strided_blocks",
        "temporal_pool",
        "use_finger_embed",
        "use_time_embed",
    ),
    "loss": ("beta_kl", "use_magnitude_weight", "weight_alpha", "weight_tau"),
    "train": (
        "epochs",
        "batch_size",
        "lr",
        "min_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "grad_clip",
        "mixed_precision",
        "seed",
        "max_steps",
    ),
    "logging": (
        "log_every",
        "val_every",
        "save_every_epoch",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "smoke_test",
    ),
}


def _load_yaml_defaults(config_path: Optional[str]) -> Dict:
    defaults = dict(_DEFAULTS)
    if not config_path:
        return defaults
    with open(config_path, "r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(payload).__name__}")
    for section, keys in _CONFIG_KEYS.items():
        values = payload.get(section, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
        unknown = sorted(set(values) - set(keys))
        if unknown:
            raise ValueError(f"Unknown keys in config section {section!r}: {unknown}")
        defaults.update(values)
    return defaults


def _build_arg_parser(defaults: Dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str, default=defaults.get("data_root"), required=defaults.get("data_root") is None)
    p.add_argument("--data_format", type=str, default=defaults["data_format"], choices=["hdf5", "lerobot", "parquet"])
    p.add_argument("--lerobot_repo_id", type=str, default=defaults["lerobot_repo_id"])
    p.add_argument("--source_window", type=int, default=defaults["source_window"])
    p.add_argument("--input_window", type=int, default=defaults["input_window"])
    p.add_argument("--subsample_stride", type=int, default=defaults["subsample_stride"])
    p.add_argument("--stride", type=int, default=defaults["stride"])
    p.add_argument("--val_ratio", type=float, default=defaults["val_ratio"])
    p.add_argument("--num_workers", type=int, default=defaults["num_workers"])

    p.add_argument("--hidden_channels", type=int, default=defaults["hidden_channels"])
    p.add_argument("--bottleneck_channels", type=int, default=defaults["bottleneck_channels"])
    p.add_argument("--bottleneck_T", type=int, default=defaults["bottleneck_T"])
    p.add_argument("--latent_dim", type=int, default=defaults["latent_dim"])
    p.add_argument("--n_strided_blocks", type=int, default=defaults["n_strided_blocks"])
    p.add_argument("--temporal_pool", type=str, default=defaults["temporal_pool"], choices=["attn", "flatten_mlp"])
    p.add_argument("--use_finger_embed", type=int, default=defaults["use_finger_embed"])
    p.add_argument("--use_time_embed", type=int, default=defaults["use_time_embed"])
    p.add_argument("--beta_kl", type=float, default=defaults["beta_kl"])
    p.add_argument("--use_magnitude_weight", type=int, default=defaults["use_magnitude_weight"])
    p.add_argument("--weight_alpha", type=float, default=defaults["weight_alpha"])
    p.add_argument("--weight_tau", type=float, default=defaults["weight_tau"])

    p.add_argument("--output_dir", type=str, default=defaults.get("output_dir"), required=defaults.get("output_dir") is None)
    p.add_argument("--run_name", type=str, default=defaults["run_name"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--min_lr_ratio", type=float, default=defaults["min_lr_ratio"])
    p.add_argument("--warmup_steps", type=int, default=defaults["warmup_steps"])
    p.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    p.add_argument("--grad_clip", type=float, default=defaults["grad_clip"])
    p.add_argument("--mixed_precision", type=str, default=defaults["mixed_precision"], choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--max_steps", type=int, default=defaults["max_steps"])

    p.add_argument("--log_every", type=int, default=defaults["log_every"])
    p.add_argument("--val_every", type=int, default=defaults["val_every"])
    p.add_argument("--save_every_epoch", type=int, default=defaults["save_every_epoch"])
    p.add_argument("--use_wandb", type=int, default=defaults["use_wandb"])
    p.add_argument("--wandb_project", type=str, default=defaults["wandb_project"])
    p.add_argument("--wandb_entity", type=str, default=defaults["wandb_entity"])
    p.add_argument("--smoke_test", type=int, default=defaults["smoke_test"])
    return p


def parse_args(argv=None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = _load_yaml_defaults(config_args.config)
    args = _build_arg_parser(defaults).parse_args(argv)
    args.config = config_args.config
    return args


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


def _init_wandb(args: argparse.Namespace, cfg: TactileVAEConfig, run_name: str) -> bool:
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={**vars(args), **cfg.to_dict()},
        )
        return True
    except Exception as e:
        print(f"[TactileVAE] wandb disabled: {e}")
        return False


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
    elif args.data_format == "lerobot":
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
    else:
        stats = TacF6Stats.from_lerobot_root(args.data_root)
        full_ds = ParquetF6ChunkDataset(
            root=args.data_root,
            source_window=args.source_window,
            input_window=args.input_window,
            subsample_stride=args.subsample_stride,
            stride=args.stride,
            stats=stats,
        )
        stats = full_ds.stats
        ep_ids = [ep_id for ep_id, _, _ in full_ds._episodes]
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(ep_ids))
        n_val_ep = max(1, int(round(len(ep_ids) * args.val_ratio))) if len(ep_ids) > 1 else 0
        if n_val_ep > 0:
            val_eps = [ep_ids[i] for i in sorted(perm[:n_val_ep].tolist())]
            tr_eps = [ep_ids[i] for i in sorted(perm[n_val_ep:].tolist())]
            train_ds = ParquetF6ChunkDataset(
                root=args.data_root,
                source_window=args.source_window,
                input_window=args.input_window,
                subsample_stride=args.subsample_stride,
                stride=args.stride,
                stats=stats,
                episodes=tr_eps,
            )
            val_ds = ParquetF6ChunkDataset(
                root=args.data_root,
                source_window=args.source_window,
                input_window=args.input_window,
                subsample_stride=args.subsample_stride,
                stride=args.stride,
                stats=stats,
                episodes=val_eps,
            )
        else:
            n_val = max(1, int(round(len(full_ds) * args.val_ratio)))
            n_train = len(full_ds) - n_val
            train_ds, val_ds = random_split(
                full_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(args.seed),
            )
            train_ds.collate_fn = full_ds.collate_fn
            val_ds.collate_fn = full_ds.collate_fn

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
        use_wandb = _init_wandb(args, cfg, run_name)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    global_step = 0
    train_examples = 0
    last_epoch = 0
    t0 = time.time()
    stop_training = False
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        for batch in train_loader:
            step_t0 = time.time()
            lr_now = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["f6"], batch["magnitude"], sample=True)
            accelerator.backward(out["total_loss"])
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step_elapsed = max(time.time() - step_t0, 1e-8)
            local_batch = int(batch["f6"].shape[0])
            global_batch = local_batch * accelerator.num_processes
            train_examples += global_batch
            samples_per_sec = global_batch / step_elapsed

            if global_step % args.log_every == 0 and accelerator.is_main_process:
                msg = (
                    f"[step {global_step:7d} | ep {epoch:2d}] "
                    f"loss={out['total_loss'].item():.4f} "
                    f"recon={out['recon_loss'].item():.4f} "
                    f"kl={out['kl_loss'].item():.4f} "
                    f"lr={lr_now:.2e} "
                    f"samples/s={samples_per_sec:.1f} "
                    f"examples={train_examples} "
                    f"elapsed={time.time() - t0:.0f}s")
                accelerator.print(msg)
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": out["total_loss"].item(),
                        "train/recon": out["recon_loss"].item(),
                        "train/kl": out["kl_loss"].item(),
                        "train/samples_per_sec": samples_per_sec,
                        "train/examples": train_examples,
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
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
            if args.smoke_test and global_step >= 5:
                stop_training = True
                break
        if stop_training:
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
        epoch=last_epoch,
    )
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()
    accelerator.print(f"[TactileVAE] Done. Total wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
