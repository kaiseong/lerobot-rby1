#!/usr/bin/env bash
set -euo pipefail

cd /home/kgs/lerobot_rby1

OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-rainbowrobotics/merged_dataset}"
SRC_A="${SRC_A:-rainbowrobotics/dataset_a}"
SRC_B="${SRC_B:-rainbowrobotics/dataset_b}"
ROOT="${ROOT:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

usage() {
  cat <<'EOF'
Usage:
  ./merge_datasets.sh [options]

Options:
  --repo-id REPO_ID      Output merged dataset repo id
  --src-a REPO_ID        First source dataset repo id
  --src-b REPO_ID        Second source dataset repo id
  --root PATH            Local dataset root override
  --push-to-hub          Push merged dataset to Hugging Face Hub
  --help                 Show this help message

Environment overrides:
  OUTPUT_REPO_ID, SRC_A, SRC_B, ROOT, PUSH_TO_HUB

Examples:
  ./merge_datasets.sh \
    --repo-id rainbowrobotics/bin_merged \
    --src-a rainbowrobotics/bin_old \
    --src-b rainbowrobotics/bin_new

  ./merge_datasets.sh \
    --repo-id rainbowrobotics/bin_merged \
    --src-a rainbowrobotics/bin_old \
    --src-b rainbowrobotics/bin_new \
    --push-to-hub
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      OUTPUT_REPO_ID="$2"
      shift 2
      ;;
    --src-a)
      SRC_A="$2"
      shift 2
      ;;
    --src-b)
      SRC_B="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --push-to-hub)
      PUSH_TO_HUB="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "merge output: ${OUTPUT_REPO_ID}"
echo "source A    : ${SRC_A}"
echo "source B    : ${SRC_B}"
echo "root        : ${ROOT:-<default HF_LEROBOT_HOME>}"
echo "push_to_hub : ${PUSH_TO_HUB}"

CMD=(
  python -m lerobot.scripts.lerobot_edit_dataset
  --repo_id "${OUTPUT_REPO_ID}"
  --operation.type merge
  --operation.repo_ids "['${SRC_A}','${SRC_B}']"
  --push_to_hub "${PUSH_TO_HUB}"
)

if [[ -n "${ROOT}" ]]; then
  CMD+=(--root "${ROOT}")
fi

PYTHONPATH=src "${CMD[@]}"
