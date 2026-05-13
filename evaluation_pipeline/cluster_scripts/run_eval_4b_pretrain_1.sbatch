#!/bin/bash
#SBATCH --job-name=eval_4b_1
#SBATCH --output=evaluate_4b_pretrain_1.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

# Sweep Part 1: 4B pretrain steps 500 to 3000
for MODEL in "4b-pretrain-step-500" "4b-pretrain-step-1000" "4b-pretrain-step-1500" "4b-pretrain-step-2000" "4b-pretrain-step-2500" "4b-pretrain-step-3000"; do
    echo ""
    echo "========================================"
    echo "[EVAL] ${MODEL} on mFollowIR"
    echo "========================================"
    python evaluation_pipeline/evaluate.py \
        --config evaluation_pipeline/configs/eval_4b_pretrain_checkpoints.yaml \
        --models "${MODEL}" \
        --datasets "mfollowir_ru"
done
