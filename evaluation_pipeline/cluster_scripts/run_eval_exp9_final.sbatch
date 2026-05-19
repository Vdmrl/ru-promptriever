#!/bin/bash
#SBATCH --job-name=eval_exp9_final
#SBATCH --output=evaluate_exp9_final.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

python evaluation_pipeline/evaluate.py \
    --config evaluation_pipeline/configs/eval_exp9_final.yaml \
    --skip-existing
