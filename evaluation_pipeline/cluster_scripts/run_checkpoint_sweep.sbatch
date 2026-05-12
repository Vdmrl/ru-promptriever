#!/bin/bash
#SBATCH --job-name=sweep_1
#SBATCH --output=evaluate_sweep_1.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

# Sweep Part 1: continues steps 60, 120, 180, 240
for MODEL in "continues-step-60" "continues-step-120" "continues-step-180" "continues-step-240"; do
    echo ""
    echo "========================================"
    echo "[EVAL] ${MODEL} on mFollowIR"
    echo "========================================"
    python evaluation_pipeline/evaluate.py \
        --config evaluation_pipeline/configs/checkpoint_sweep.yaml \
        --models "${MODEL}" \
        --datasets "mfollowir_ru"
done
