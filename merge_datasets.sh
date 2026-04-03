#!/usr/bin/env bash
set -euo pipefail


OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-rainbowrobotics/simtos_0402_merge}"
SRC_A="${SRC_A:-rainbowrobotics/simtos_final_one_item_can_rotate}"
SRC_B="${SRC_B:-rainbowrobotics/simtos_final_one_item_can}"
SRC_REPO_IDS="${SRC_REPO_IDS:-}"
ROOT="${ROOT:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

SOURCE_REPO_IDS=()

if [[ -n "${SRC_REPO_IDS}" ]]; then
  IFS=',' read -r -a SOURCE_REPO_IDS <<< "${SRC_REPO_IDS}"
fi

usage() {
  cat <<'EOF'
Usage:
  ./merge_datasets.sh [options]

Options:
  --repo-id REPO_ID      Output merged dataset repo id
  --src REPO_ID          Source dataset repo id (repeatable)
  --src-a REPO_ID        First source dataset repo id
  --src-b REPO_ID        Second source dataset repo id
  --root PATH            Local dataset root override
  --push-to-hub          Push merged dataset to Hugging Face Hub
  --help                 Show this help message

Environment overrides:
  OUTPUT_REPO_ID, SRC_REPO_IDS, SRC_A, SRC_B, ROOT, PUSH_TO_HUB

Examples:
  ./merge_datasets.sh \
    --repo-id rainbowrobotics/bin_merged \
    --src rainbowrobotics/bin_old \
    --src rainbowrobotics/bin_new

  ./merge_datasets.sh \
    --repo-id rainbowrobotics/bin_merged \
    --src rainbowrobotics/bin_old \
    --src rainbowrobotics/bin_new \
    --src rainbowrobotics/bin_more \
    --push-to-hub

  SRC_REPO_IDS=rainbowrobotics/bin_old,rainbowrobotics/bin_new ./merge_datasets.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      OUTPUT_REPO_ID="$2"
      shift 2
      ;;
    --src)
      SOURCE_REPO_IDS+=("$2")
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

if [[ "${#SOURCE_REPO_IDS[@]}" -eq 0 ]]; then
  SOURCE_REPO_IDS=("${SRC_A}" "${SRC_B}")
fi

if [[ "${#SOURCE_REPO_IDS[@]}" -lt 2 ]]; then
  echo "At least two source datasets are required." >&2
  usage
  exit 1
fi

REPO_IDS_LITERAL="["
for i in "${!SOURCE_REPO_IDS[@]}"; do
  if [[ "${i}" -gt 0 ]]; then
    REPO_IDS_LITERAL+=","
  fi
  REPO_IDS_LITERAL+="'${SOURCE_REPO_IDS[$i]}'"
done
REPO_IDS_LITERAL+="]"

echo "merge output: ${OUTPUT_REPO_ID}"
printf 'sources     : %s\n' "${SOURCE_REPO_IDS[*]}"
echo "root        : ${ROOT:-<default HF_LEROBOT_HOME>}"
echo "push_to_hub : ${PUSH_TO_HUB}"

CMD=(
  python -m lerobot.scripts.lerobot_edit_dataset
  --repo_id "${OUTPUT_REPO_ID}"
  --operation.type merge
  --operation.repo_ids "${REPO_IDS_LITERAL}"
  --push_to_hub "${PUSH_TO_HUB}"
)

if [[ -n "${ROOT}" ]]; then
  CMD+=(--root "${ROOT}")
fi

PYTHONPATH=src "${CMD[@]}"
