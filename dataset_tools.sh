#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
TOOLS_DIR="$REPO_DIR/tools"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONPATH="$REPO_DIR/src:$TOOLS_DIR${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  ./dataset_tools.sh trim  [trim_stationary_dataset.py args...]
  ./dataset_tools.sh merge [merge_datasets.py args...]

Examples:
  ./dataset_tools.sh trim \
    --repo-id rainbowrobotics/source \
    --new-repo-id rainbowrobotics/source_trimmed \
    --workers auto \
    --dry-run

  ./dataset_tools.sh merge \
    --source rainbowrobotics/a \
    --source rainbowrobotics/b \
    --new-repo-id rainbowrobotics/merged \
    --dry-run

Notes:
  - Input roots are optional when datasets are in the LeRobot cache or can be downloaded.
  - Output root defaults to $HF_LEROBOT_HOME/<new-repo-id>.
  - trim is logical-only: it does not decode, physically trim, or re-encode videos.
  - merge defaults to no remux/re-encode; videos are copied.
  - Set PYTHON=/path/to/python if you do not want to use python3.
  - Add --push-to-hub only when you actually want to upload.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command="$1"
shift

case "$command" in
  trim|trim-stationary|trim_stationary)
    exec "$PYTHON_BIN" "$TOOLS_DIR/trim_stationary_dataset.py" "$@"
    ;;
  merge)
    exec "$PYTHON_BIN" "$TOOLS_DIR/merge_datasets.py" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
