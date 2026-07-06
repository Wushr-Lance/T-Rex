# Tactile Force Filtering Visualization

This workflow helps choose a tactile-force filter before training the
hand-wise tactile VAE. It follows the training unit exactly:

```text
raw tactile frame:          [10, 6]
64-frame bimanual window:   [64, 10, 6]
split into two hand windows [64, 5, 6]
```

Filtering is evaluated on the contiguous raw 64-frame hand window. The VAE
can still subsample that kept window to `[16, 5, 6]` later.

## Force Metric

The scripts use the 3D force norm per finger:

```text
finger_force = sqrt(Fx^2 + Fy^2 + Fz^2)
```

The moment channels `Mx, My, Mz` are ignored for filtering. Thresholds are
therefore in the same units as the dataset's force channels.

## Setup

From the repository root:

```bash
cd /home/shenrui/egoscale2/T-Rex
```

Use the project environment. With `uv`:

```bash
uv run python - <<'PY'
import numpy, pandas, pyarrow, matplotlib
print("filter visualization dependencies ok")
PY
```

If you use conda instead, activate the environment that can already run
`tactile_vae/train.py`.

## First Pass: Auto Thresholds

Run the default auto-threshold sweep:

```bash
PYTHONPATH=$PWD uv run python -m tactile_vae.scripts.visualize_tactile_force_filters \
  --data_root /data/d3/shenrui/trex_dataset_no_videos \
  --output_dir tactile_vae/outputs/filter_viz/trex_default \
  --source_window 64 \
  --stride 4 \
  --thresholds auto \
  --threshold_percentiles 50,60,70,80,90,95,97,99 \
  --min_active_frames 1 \
  --min_consecutive_frames 8 \
  --example_episodes 8 \
  --seed 42
```

For a quick smoke run before scanning the full dataset, add:

```bash
--max_episodes 20 --example_episodes 3
```

The script writes:

```text
tactile_vae/outputs/filter_viz/trex_default/
├── filter_summary.csv
├── continuity_summary.csv
├── threshold_candidates.json
├── run_config.json
├── force_distributions.png
├── threshold_sweep.png
├── continuity_sweep.png
└── examples/
```

## How To Read The Outputs

`threshold_candidates.json` contains percentile suggestions from the first
pass. When `--thresholds auto` is used, the selected thresholds come from the
`any_finger` score percentiles.

`force_distributions.png` shows histograms and CDFs for:

- `any_finger`: maximum single-finger force in the 64-frame hand window.
- `mean5`: maximum frame-wise average over the 5 fingers.
- `window_mean5`: average over all `64 x 5` force values.

`threshold_sweep.png` shows the kept hand-window ratio for each method,
threshold, and hand. Use this to find thresholds that remove blank windows
without collapsing the dataset size.

`continuity_sweep.png` shows how many windows remain as you require more
active raw frames. This is important because training samples contiguous
64-frame windows, not isolated frames.

`examples/*.png` shows selected episode timelines. Green spans are accepted
windows for the example method/threshold; red spans are rejected windows.

## Filtering Methods

`any_finger`:

```text
Pass if at least min_active_frames raw frames have any finger >= threshold.
```

This is the loosest method. With `--min_active_frames 1`, one force event
anywhere inside the 64-frame window is enough.

`mean5`:

```text
Pass if at least min_active_frames raw frames have average(5 fingers) >= threshold.
```

This favors windows where force is spread across multiple fingers.

`sustained_any`:

```text
Pass if any finger stays >= threshold for min_consecutive_frames consecutive frames.
```

This is useful when you want sustained contact rather than a single spike.

`window_mean5`:

```text
Pass if average over all 64 x 5 force values >= threshold.
```

This is the strictest method and is sensitive to long blank periods inside
the 64-frame window.

## Refining Thresholds With Your Mentor

After reviewing the first pass, rerun with explicit thresholds:

```bash
PYTHONPATH=$PWD uv run python -m tactile_vae.scripts.visualize_tactile_force_filters \
  --data_root /data/d3/shenrui/trex_dataset_no_videos \
  --output_dir tactile_vae/outputs/filter_viz/trex_refined \
  --thresholds 0.01,0.02,0.05,0.1,0.2,0.5 \
  --min_active_frames 4 \
  --min_consecutive_frames 8 \
  --example_episodes 12
```

Tune these hyperparameters:

- `--thresholds`: force thresholds to compare.
- `--min_active_frames`: required number of raw frames for `any_finger` and `mean5`.
- `--min_consecutive_frames`: required consecutive active raw frames for `sustained_any`.
- `--stride`: candidate window spacing. Use the same value as training for final estimates.
- `--source_window`: contiguous raw-frame window length. Keep `64` for current VAE training.
- `--example_method`: method used for green/red episode timeline overlays.
- `--example_threshold`: threshold used for examples. If omitted, the middle selected threshold is used.
- `--example_max_frames`: maximum raw frames shown per example episode.

For mentor discussion, compare:

- Kept ratio in `filter_summary.csv`.
- Left/right balance in `threshold_sweep.png`.
- Whether accepted spans in `examples/` cover real contact and reject blank periods.
- Whether continuity settings still keep enough windows for training.

## Optional Window Index CSVs

By default, kept-window CSVs are disabled because the full dataset can produce
large files. To export them:

```bash
PYTHONPATH=$PWD uv run python -m tactile_vae.scripts.visualize_tactile_force_filters \
  --data_root /data/d3/shenrui/trex_dataset_no_videos \
  --output_dir tactile_vae/outputs/filter_viz/trex_indexed \
  --thresholds 0.05 \
  --save_window_indices 1
```

This creates files like:

```text
window_index_any_finger_0p05.csv
window_index_mean5_0p05.csv
window_index_sustained_any_0p05.csv
window_index_window_mean5_0p05.csv
```

Each row is one kept hand-wise training candidate:

```text
episode_index, frame_start, hand, hand_name, threshold, method, score
```

These CSVs are not used by training yet. They are intended as reusable
artifacts if you later decide to integrate a final filter into the dataset
loader.

## Recommended Decision Path

1. Run a quick smoke pass with `--max_episodes 20`.
2. Run the full auto-threshold pass.
3. Pick 4-8 explicit thresholds around the useful region in the sweep plot.
4. Compare `any_finger`, `mean5`, and `sustained_any` with continuity settings.
5. Choose the final method and hyperparameters before modifying training.
