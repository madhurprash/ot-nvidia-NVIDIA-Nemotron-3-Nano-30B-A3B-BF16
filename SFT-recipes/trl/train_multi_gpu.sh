#!/bin/bash
# Multi-GPU training launch script for TRL with DDP
# This script uses torchrun to launch distributed training across all available GPUs

# Number of GPUs to use (default: 8, all available GPUs)
NGPUS=${1:-8}

# Config file path (default: train.yaml)
CONFIG_FILE=${2:-train.yaml}

echo "=========================================="
echo "Multi-GPU Training with TRL DDP"
echo "=========================================="
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Launch distributed training using torchrun
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    train_sft.py $CONFIG_FILE

echo "=========================================="
echo "Training completed!"
echo "=========================================="
