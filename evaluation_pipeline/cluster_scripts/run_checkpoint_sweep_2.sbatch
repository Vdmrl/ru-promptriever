#!/bin/bash
#SBATCH --job-name=sweep_2
#SBATCH --output=evaluate_sweep_2.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

# Sweep Part 2: continues steps 300, 360, 420
for MODEL in "continues-step-300" "continues-step-360" "continues-step-420"; do
    echo ""
    echo "========================================"
    echo "[EVAL] ${MODEL} on mFollowIR"
    echo "========================================"
    python evaluation_pipeline/evaluate.py \
        --config evaluation_pipeline/configs/checkpoint_sweep.yaml \
        --models "${MODEL}" \
        --datasets "mfollowir_ru"
done
