#!/bin/bash
#SBATCH --job-name=sweep_4
#SBATCH --output=evaluate_sweep_4.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

# Sweep Part 4: gold-mix steps 300, 360, 420, 480
for MODEL in "gold-mix-step-300" "gold-mix-step-360" "gold-mix-step-420" "gold-mix-step-480"; do
    echo ""
    echo "========================================"
    echo "[EVAL] ${MODEL} on mFollowIR"
    echo "========================================"
    python evaluation_pipeline/evaluate.py \
        --config evaluation_pipeline/configs/checkpoint_sweep.yaml \
        --models "${MODEL}" \
        --datasets "mfollowir_ru"
done
