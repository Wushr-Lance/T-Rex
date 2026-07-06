#!/bin/bash
# Train the hand-wise tactile VAE on T-Rex dataset.
#
# Runtime overrides:
#   CONFIG=<repo>/tactile_vae/config/tactile_vae_trex_parquet.yaml
#   CUDA_VISIBLE_DEVICES=0,1
#   USE_WANDB=1, WANDB_MODE=online
#   WANDB_ENTITY=berkeley_bair, WANDB_PROJECT=trex_tactile_vae
#   RUN_NAME=tactile_vae_trex_parquet_2gpu
#   OUTPUT_DIR=<repo>/tactile_vae/outputs
#   TRITON_CACHE_DIR=/data/d3/shenrui/triton_cache
#
# Training hyperparameters such as batch size, learning rate, epochs, window
# size, model size, workers, and validation cadence live in CONFIG.

set -euo pipefail

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PARENT_DIR}"

: "${CONFIG:=${PARENT_DIR}/tactile_vae/config/tactile_vae_trex_parquet.yaml}"
: "${USE_WANDB:=0}"
: "${WANDB_API_KEY:=}"
: "${WANDB_ENTITY:=berkeley_bair}"
: "${WANDB_PROJECT:=trex_tactile_vae}"
if [ -z "${WANDB_MODE:-}" ]; then
    if [ "${USE_WANDB}" = "1" ]; then
        WANDB_MODE=online
    else
        WANDB_MODE=offline
    fi
fi
: "${CUDA_VISIBLE_DEVICES:=0}"

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES WANDB_API_KEY WANDB_MODE
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH:-}"

LOG_ROOT="${OUTPUT_DIR:-${PARENT_DIR}/tactile_vae/outputs}"
LOG_NAME="${RUN_NAME:-tactile_vae_$(date +%m%d_%H%M)}"
LOCAL_LOG_DIR="${LOG_ROOT}/logs"
mkdir -p "${LOCAL_LOG_DIR}"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${LOG_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Mirroring training output to ${LOCAL_LOG_FILE}"
echo ">>> Using ${N_GPUS} GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo ">>> config     = ${CONFIG}"
echo ">>> output_dir = ${OUTPUT_DIR:-from config}"
echo ">>> run_name   = ${RUN_NAME:-from config}"
echo ">>> wandb      = mode=${WANDB_MODE} entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

CMD=(
accelerate launch
    --num_processes "${N_GPUS}" \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    -m tactile_vae.train \
    --config            "${CONFIG}" \
    --use_wandb          "${USE_WANDB}" \
    --wandb_project      "${WANDB_PROJECT}" \
    --wandb_entity       "${WANDB_ENTITY}"
)

if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${RUN_NAME:-}" ]; then
    CMD+=(--run_name "${RUN_NAME}")
fi

"${CMD[@]}" 2>&1 | tee -a "${LOCAL_LOG_FILE}"

if [ -n "${OUTPUT_DIR:-}" ] && [ -n "${RUN_NAME:-}" ]; then
    echo ">>> Done. Checkpoint: ${OUTPUT_DIR}/${RUN_NAME}/latest.pt"
else
    echo ">>> Done. Check the configured output_dir/run_name for latest.pt"
fi
