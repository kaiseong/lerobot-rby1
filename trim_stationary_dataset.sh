#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ID="rainbowrobotics/bin_0318_19_merged"
NEW_REPO_ID="kaiseong/bin_0318-19_merged_trim_0.2"
PUSH_TO_HUB="true"
KEEP_START_SECONDS="0.2"
KEEP_END_SECONDS="0.2"
STATE_KEY="observation.state"
STATE_EPSILON="0.009"

usage() {
  cat <<EOF
Usage: ./trim_stationary_dataset.sh [options]

Options:
  --repo-id <repo_id>
  --new-repo-id <repo_id>
  --push-to-hub <true|false>
  --keep-start-seconds <seconds>
  --keep-end-seconds <seconds>
  --state-key <feature_key>
  --state-epsilon <epsilon>
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --new-repo-id)
      NEW_REPO_ID="$2"
      shift 2
      ;;
    --push-to-hub)
      PUSH_TO_HUB="$2"
      shift 2
      ;;
    --keep-start-seconds)
      KEEP_START_SECONDS="$2"
      shift 2
      ;;
    --keep-end-seconds)
      KEEP_END_SECONDS="$2"
      shift 2
      ;;
    --state-key)
      STATE_KEY="$2"
      shift 2
      ;;
    --state-epsilon)
      STATE_EPSILON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cmd=(
  python -m lerobot.scripts.lerobot_edit_dataset
  --repo_id "$REPO_ID"
  --push_to_hub "$PUSH_TO_HUB"
  --operation.type trim_stationary_episode_edges
  --operation.keep_start_seconds "$KEEP_START_SECONDS"
  --operation.keep_end_seconds "$KEEP_END_SECONDS"
  --operation.state_key "$STATE_KEY"
  --operation.state_epsilon "$STATE_EPSILON"
)

if [[ -n "$NEW_REPO_ID" ]]; then
  cmd+=(--new_repo_id "$NEW_REPO_ID")
fi

printf 'repo_id=%s
new_repo_id=%s
push_to_hub=%s
keep_start_seconds=%s
keep_end_seconds=%s
state_key=%s
state_epsilon=%s
'   "$REPO_ID" "$NEW_REPO_ID" "$PUSH_TO_HUB" "$KEEP_START_SECONDS" "$KEEP_END_SECONDS" "$STATE_KEY" "$STATE_EPSILON"

PYTHONPATH=src "${cmd[@]}"
