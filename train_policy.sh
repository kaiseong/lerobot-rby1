#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./train_policy.sh --policy POLICY_TYPE --dataset DATASET_REPO_ID [options] [-- extra lerobot-train args...]

Supported policy types:
  act
  diffusion
  groot
  pi05      (alias: pi0.5)

Examples:
  ./train_policy.sh \
    --policy act \
    --dataset rainbowrobotics/bin_0318_19_merged_v2 \
    --output-dir ./outputs/train/act_run \
    --batch-size 8 \
    --steps 200000 \
    --val-ratio 0.1

  ./train_policy.sh \
    --policy diffusion \
    --dataset rainbowrobotics/bin_0318_19_merged_v2 \
    --batch-size 16 \
    --steps 100000 \
    --val-ratio 0.1

  ./train_policy.sh \
    --policy groot \
    --dataset rainbowrobotics/bin_0318_19_merged_v2 \
    --enable-crop \
    --crop-resize-size "[480,480]" \
    --crop-params '{"observation.images.front":[0,80,480,480],"observation.images.right":[160,0,480,480]}'

  ./train_policy.sh \
    --policy pi05 \
    --dataset rainbowrobotics/bin_0318_19_merged_v2 \
    --policy-path your/pi05_checkpoint_or_repo \
    --val-ratio 0.1

Common options:
  --policy POLICY_TYPE
  --dataset DATASET_REPO_ID
  --output-dir PATH
  --job-name NAME
  --batch-size N
  --steps N
  --save-freq N
  --log-freq N
  --val-ratio FLOAT
  --val-freq N
  --num-workers N
  --device DEVICE
  --policy-path PATH_OR_REPO
  --policy-repo-id REPO_ID
  --push-to-hub
  --wandb-enable
  --wandb-project NAME
  --wandb-entity NAME
  --enable-crop
  --crop-resize-size "[H,W]"
  --crop-params JSON
  --num-processes N
  --mixed-precision MODE
  --help

Notes:
  - Validation split is episode-level. --val-ratio 0.1 means 10% of episodes go to validation.
  - If --val-freq is omitted, lerobot-train uses save_freq as the validation cadence.
  - --policy-path maps to --policy.path for fine-tuning a pretrained checkpoint.
  - For policy-specific overrides, pass them after `--`.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

POLICY_TYPE=""
DATASET_REPO_ID=""
OUTPUT_DIR=""
JOB_NAME=""
POLICY_PATH=""
POLICY_REPO_ID=""

BATCH_SIZE=8
STEPS=100000
SAVE_FREQ=20000
LOG_FREQ=1000
VAL_RATIO=0.0
VAL_FREQ=""
NUM_WORKERS=4
DEVICE="cuda"

SAVE_CHECKPOINT=true
PUSH_TO_HUB=false
WANDB_ENABLE=false
WANDB_DISABLE_ARTIFACT=true
WANDB_PROJECT="lerobot"
WANDB_ENTITY=""

ENABLE_CROP=false
CROP_RESIZE_SIZE="[480,480]"
CROP_PARAMS='{"observation.images.front":[0,80,480,480],"observation.images.right":[160,0,480,480]}'

NUM_PROCESSES=""
MIXED_PRECISION=""

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      POLICY_TYPE="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET_REPO_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --job-name)
      JOB_NAME="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --steps)
      STEPS="${2:-}"
      shift 2
      ;;
    --save-freq)
      SAVE_FREQ="${2:-}"
      shift 2
      ;;
    --log-freq)
      LOG_FREQ="${2:-}"
      shift 2
      ;;
    --val-ratio)
      VAL_RATIO="${2:-}"
      shift 2
      ;;
    --val-freq)
      VAL_FREQ="${2:-}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --policy-path)
      POLICY_PATH="${2:-}"
      shift 2
      ;;
    --policy-repo-id)
      POLICY_REPO_ID="${2:-}"
      shift 2
      ;;
    --push-to-hub)
      PUSH_TO_HUB=true
      shift
      ;;
    --wandb-enable)
      WANDB_ENABLE=true
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="${2:-}"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="${2:-}"
      shift 2
      ;;
    --enable-crop)
      ENABLE_CROP=true
      shift
      ;;
    --crop-resize-size)
      CROP_RESIZE_SIZE="${2:-}"
      shift 2
      ;;
    --crop-params)
      CROP_PARAMS="${2:-}"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="${2:-}"
      shift 2
      ;;
    --mixed-precision)
      MIXED_PRECISION="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${POLICY_TYPE}" ]]; then
  echo "--policy is required" >&2
  usage
  exit 1
fi

if [[ -z "${DATASET_REPO_ID}" ]]; then
  echo "--dataset is required" >&2
  usage
  exit 1
fi

case "${POLICY_TYPE}" in
  pi0.5)
    POLICY_TYPE="pi05"
    ;;
  act|diffusion|groot|pi05)
    ;;
  *)
    echo "Unsupported policy type: ${POLICY_TYPE}" >&2
    exit 1
    ;;
esac

if [[ "${PUSH_TO_HUB}" == "true" && -z "${POLICY_REPO_ID}" ]]; then
  echo "--policy-repo-id is required when --push-to-hub is enabled" >&2
  exit 1
fi

if [[ "${POLICY_TYPE}" == "pi05" && -z "${POLICY_PATH}" ]]; then
  echo "pi05 training usually requires --policy-path pointing to a pretrained checkpoint or repo." >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="./outputs/train/${POLICY_TYPE}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="${POLICY_TYPE}_train"
fi

ACCELERATE_CMD=(accelerate launch)
if [[ -n "${NUM_PROCESSES}" ]]; then
  ACCELERATE_CMD+=(--num_processes "${NUM_PROCESSES}")
fi
if [[ -n "${MIXED_PRECISION}" ]]; then
  ACCELERATE_CMD+=(--mixed_precision "${MIXED_PRECISION}")
fi
ACCELERATE_CMD+=("$(which lerobot-train)")

TRAIN_CMD=(
  "${ACCELERATE_CMD[@]}"
  --output_dir="${OUTPUT_DIR}"
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --policy.type="${POLICY_TYPE}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.val_ratio="${VAL_RATIO}"
  --num_workers="${NUM_WORKERS}"
  --policy.device="${DEVICE}"
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
  --wandb.project="${WANDB_PROJECT}"
  --job_name="${JOB_NAME}"
)

if [[ -n "${VAL_FREQ}" ]]; then
  TRAIN_CMD+=(--val_freq="${VAL_FREQ}")
fi

if [[ -n "${POLICY_PATH}" ]]; then
  TRAIN_CMD+=(--policy.path="${POLICY_PATH}")
fi

if [[ "${PUSH_TO_HUB}" == "true" ]]; then
  TRAIN_CMD+=(
    --policy.push_to_hub=true
    --policy.repo_id="${POLICY_REPO_ID}"
  )
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_CMD+=(--wandb.entity="${WANDB_ENTITY}")
fi

if [[ "${ENABLE_CROP}" == "true" ]]; then
  TRAIN_CMD+=(
    --dataset.crop.enable=true
    --dataset.crop.resize_size="${CROP_RESIZE_SIZE}"
    --dataset.crop.params="${CROP_PARAMS}"
  )
fi

case "${POLICY_TYPE}" in
  groot)
    TRAIN_CMD+=(
      --policy.image_size="[224,224]"
      --policy.tune_visual=false
      --policy.tune_llm=false
      --policy.tune_projector=true
      --policy.tune_diffusion_model=true
      --optimizer.lr=0.0001
      --optimizer.weight_decay=1e-05
      --policy.warmup_ratio=0.05
      --dataset.image_transforms.enable=true
    )
    ;;
  act)
    TRAIN_CMD+=(
      --dataset.image_transforms.enable=true
    )
    ;;
  diffusion)
    TRAIN_CMD+=(
      --dataset.image_transforms.enable=true
    )
    ;;
  pi05)
    TRAIN_CMD+=(
      --dataset.image_transforms.enable=true
    )
    ;;
esac

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Training command:"
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'

exec "${TRAIN_CMD[@]}"
