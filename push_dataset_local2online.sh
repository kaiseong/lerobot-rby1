#!/usr/bin/env bash
set -euo pipefail

cd /home/kgs/lerobot

PYTHONPATH=src python -u - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "rainbowrobotics/bin_0318_19_merged_v2"
root = "/home/kgs/.cache/huggingface/lerobot/kaiseong/bin_0318_19_merged_crop480_trim2s"

print(f"start push: {repo_id}")
LeRobotDataset(repo_id, root=root).push_to_hub()
print("done")
PY
