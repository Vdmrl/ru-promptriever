#!/bin/bash
#SBATCH --job-name=ev_exp9
#SBATCH --output=evaluate_exp9.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

python evaluation_pipeline/evaluate.py \
    --config evaluation_pipeline/configs/baseline_qwen3-4b.yaml \
    --models "ru-promptriever-exp9-en-only" \
    --datasets "synthetic_test" "mfollowir_ru"
