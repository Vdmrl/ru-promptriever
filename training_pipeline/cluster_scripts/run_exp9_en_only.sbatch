#!/bin/bash
#SBATCH --job-name=exp9_abl
#SBATCH --output=train_exp9.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=8

module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

torchrun --nproc_per_node=8 --master_port=29509 training_pipeline/train.py --config training_pipeline/configs/exp9_en_only_ablation.yaml
