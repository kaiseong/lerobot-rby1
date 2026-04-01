#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SERVER_ADDRESS="${SERVER_ADDRESS:-192.168.0.3:5555}"
ROBOT_ADDRESS="${ROBOT_ADDRESS:-192.168.0.10:50051}"
TASK="${TASK:-pick and place}"

FPS="${FPS:-15}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
ZMQ_TIMEOUT_MS="${ZMQ_TIMEOUT_MS:-5000}"

GROOT_FRONT_CAMERA_KEY="${GROOT_FRONT_CAMERA_KEY:-front}"
GROOT_LEFT_WRIST_CAMERA_KEY="${GROOT_LEFT_WRIST_CAMERA_KEY:-left}"
GROOT_RIGHT_WRIST_CAMERA_KEY="${GROOT_RIGHT_WRIST_CAMERA_KEY:-right}"
GROOT_IMAGE_SIZE="${GROOT_IMAGE_SIZE:-[480,640]}"

IMAGE_CROP_PARAMS="${IMAGE_CROP_PARAMS:-{}}"

PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m lerobot.async_inference.robot_client \
  --backend=groot_n16_zmq \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=rby1 \
  --robot.address="${ROBOT_ADDRESS}" \
  --robot.use_torso=false \
  --robot.observe_torso=true \
  --robot.use_right_arm=true \
  --robot.use_left_arm=true \
  --robot.use_gripper=true \
  --task="${TASK}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --fps="${FPS}" \
  --client_device="${CLIENT_DEVICE}" \
  --zmq_timeout_ms="${ZMQ_TIMEOUT_MS}" \
  --groot_front_camera_key="${GROOT_FRONT_CAMERA_KEY}" \
  --groot_left_wrist_camera_key="${GROOT_LEFT_WRIST_CAMERA_KEY}" \
  --groot_right_wrist_camera_key="${GROOT_RIGHT_WRIST_CAMERA_KEY}" \
  --groot_image_size="${GROOT_IMAGE_SIZE}" \
  --image_crop_params="${IMAGE_CROP_PARAMS}"
