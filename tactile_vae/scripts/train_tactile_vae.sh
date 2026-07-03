#!/bin/bash
# Train the hand-wise tactile VAE on a T-Rex merged HDF5 root.
#
# Required:
#   DATA_ROOT=/path/to/merged_midtrain_root
#
# Optional:
#   OUTPUT_DIR=<repo>/outputs/tactile_vae
#   RUN_NAME=tactile_vae_w64to16_attn_...
#   TEMPORAL_POOL=attn|flatten_mlp
#   USE_FINGER_EMBED=1, LATENT=256, BATCH=256, EPOCHS=30, LR=3e-4

set -euo pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

: "${DATA_ROOT:?set DATA_ROOT to the merged T-Rex data root}"
: "${OUTPUT_DIR:=${PARENT_DIR}/outputs/tactile_vae}"
: "${SOURCE_WINDOW:=64}"
: "${INPUT_WINDOW:=16}"
: "${SUBSAMPLE_STRIDE:=4}"
: "${STRIDE:=4}"
: "${TEMPORAL_POOL:=attn}"
: "${USE_FINGER_EMBED:=1}"
: "${LATENT:=256}"
: "${BATCH:=256}"
: "${EPOCHS:=30}"
: "${LR:=3e-4}"
: "${RUN_NAME:=tactile_vae_w${SOURCE_WINDOW}to${INPUT_WINDOW}_${TEMPORAL_POOL}_z${LATENT}_$(date +%m%d_%H%M)}"
: "${USE_WANDB:=0}"
: "${WANDB_API_KEY:=}"
: "${WANDB_MODE:=${WANDB_API_KEY:+online}}"
: "${WANDB_MODE:=offline}"
: "${CUDA_VISIBLE_DEVICES:=0}"

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES WANDB_API_KEY WANDB_MODE
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH:-}"

LOCAL_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOCAL_LOG_DIR}"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Mirroring training output to ${LOCAL_LOG_FILE}"
echo ">>> Using ${N_GPUS} GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo ">>> data_root  = ${DATA_ROOT}"
echo ">>> output_dir = ${OUTPUT_DIR}/${RUN_NAME}"

accelerate launch \
    --num_processes "${N_GPUS}" \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    -m tactile_vae.train \
    --data_root          "${DATA_ROOT}" \
    --output_dir         "${OUTPUT_DIR}" \
    --run_name           "${RUN_NAME}" \
    --source_window      "${SOURCE_WINDOW}" \
    --input_window       "${INPUT_WINDOW}" \
    --subsample_stride   "${SUBSAMPLE_STRIDE}" \
    --stride             "${STRIDE}" \
    --temporal_pool      "${TEMPORAL_POOL}" \
    --use_finger_embed   "${USE_FINGER_EMBED}" \
    --latent_dim         "${LATENT}" \
    --epochs             "${EPOCHS}" \
    --batch_size         "${BATCH}" \
    --lr                 "${LR}" \
    --num_workers        4 \
    --val_every          2000 \
    --use_wandb          "${USE_WANDB}" \
    2>&1 | tee -a "${LOCAL_LOG_FILE}"

echo ">>> Done. Checkpoint: ${OUTPUT_DIR}/${RUN_NAME}/latest.pt"

