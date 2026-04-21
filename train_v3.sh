#!/bin/bash

# Train full_v3 experiment (old config with VastTrack)

set -e

cd /root/user-data/wyp/ATCTrack_align

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled

PYTHON=/root/user-data/envs/wyp_vlt/bin/python

echo "=========================================="
echo "Training atctrack_base_semantic_full_v3"
echo "Started: $(date)"
echo "=========================================="

$PYTHON lib/train/run_training.py \
  --script atctrack \
  --config atctrack_base_semantic_full_v3 \
  --save_dir /root/user-data/wyp/ATCTrack_align/output \
  --use_lmdb 0 \
  --use_wandb 0

echo ""
echo "=========================================="
echo "Training completed: $(date)"
echo "=========================================="
