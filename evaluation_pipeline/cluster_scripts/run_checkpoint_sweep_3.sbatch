#!/bin/bash
#SBATCH --job-name=sweep_3
#SBATCH --output=evaluate_sweep_3.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

# Sweep Part 3: gold-mix steps 60, 120, 180, 240
for MODEL in "gold-mix-step-60" "gold-mix-step-120" "gold-mix-step-180" "gold-mix-step-240"; do
    echo ""
    echo "========================================"
    echo "[EVAL] ${MODEL} on mFollowIR"
    echo "========================================"
    python evaluation_pipeline/evaluate.py \
        --config evaluation_pipeline/configs/checkpoint_sweep.yaml \
        --models "${MODEL}" \
        --datasets "mfollowir_ru"
done
