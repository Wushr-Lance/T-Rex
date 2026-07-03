"""Download a tiny T-Rex LeRobot subset for tactile VAE sanity checks.

This pulls metadata and the trajectory parquet files for selected episodes,
without videos. It is intentionally capped so it never downloads the full
dataset by accident.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve()
_PARENT = _THIS_DIR.parents[2]
_QUICKSTART_SRC = _PARENT / "dataset_quickstart" / "src"
if str(_QUICKSTART_SRC) not in sys.path:
    sys.path.insert(0, str(_QUICKSTART_SRC))

from trex_dataset_quickstart import hub  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="zekaiwang/trex_dataset")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--max_gb", type=float, default=0.2)
    p.add_argument("--revision", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "trex_tactile_vae_sanity")
    root = hub.meta_root(args.repo_id, cache_dir=cache_dir, revision=args.revision)
    print(f"metadata root: {root}")
    for ep in args.episodes:
        local = hub.fetch_episode(
            args.repo_id,
            episode_index=ep,
            cache_dir=cache_dir,
            max_gb=args.max_gb,
            include_videos=False,
            revision=args.revision,
        )
        print(f"episode {ep} available under: {local}")


if __name__ == "__main__":
    main()
