#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch $(which lerobot-train) \
  --output_dir=./outputs/train/bin_0318_merged_1to3_v3_crop_groot_N15_20k \
  --save_checkpoint=true \
  --batch_size=16 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=1000 \
  --policy.push_to_hub=true \
  --policy.type=groot \
  --policy.repo_id="rainbowrobotics/groot_N15_20k_v2" \
  --policy.image_size="[224,224]" \
  --dataset.repo_id="rainbowrobotics/bin_0318_19_merged_v2" \
  --dataset.crop.enable=true \
  --dataset.crop.resize_size="[480,480]" \
  --dataset.crop.params='{"observation.images.front":[0,80,480,480],"observation.images.right":[160,0,480,480]}' \
  --policy.tune_visual=false \
  --policy.tune_llm=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --optimizer.lr=0.0001 \
  --optimizer.weight_decay=1e-05 \
  --policy.warmup_ratio=0.05 \
  --dataset.image_transforms.enable=true \
  --num_workers=2 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name="rby1_groot_N15_FT"
