#!/bin/bash
# TAHPNet 消融实验脚本
#
# 实验设计：
# 1. baseline: 无时序模块（单帧估计）
# 2. full: 完整 TAHPNet（双向GRU + 2层）
# 3. no_bidir: 单向 GRU
# 4. shallow: 单层 GRU

set -e

DATA_ROOT="data/tracked_output"
POSE_LABELS="data/pose_output/all_poses.json"
BASE_DIR="checkpoints/tahpnet_ablation"
EPOCHS=30
BATCH_SIZE=8
SEQ_LEN=16

echo "======================================"
echo "TAHPNet 消融实验"
echo "======================================"

# 1. Baseline (无时序模块)
echo ""
echo "[1/4] 训练 Baseline..."
python3 experiments/scripts/train_tahpnet.py \
    --data-root $DATA_ROOT \
    --pose-labels $POSE_LABELS \
    --save-dir ${BASE_DIR}/baseline \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --no-temporal \
    --device cuda:0

# 2. Full TAHPNet
echo ""
echo "[2/4] 训练 Full TAHPNet..."
python3 experiments/scripts/train_tahpnet.py \
    --data-root $DATA_ROOT \
    --pose-labels $POSE_LABELS \
    --save-dir ${BASE_DIR}/full \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --device cuda:0

echo ""
echo "消融实验完成! 结果: ${BASE_DIR}"
