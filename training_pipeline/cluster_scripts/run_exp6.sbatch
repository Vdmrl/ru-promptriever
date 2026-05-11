#!/bin/bash
#SBATCH --job-name=exp6_06b
#SBATCH --output=train_exp6.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16


module load Python/PyTorch_GPU_v2.4
export WANDB_MODE=disabled

torchrun --nproc_per_node=4 --master_port=29506 training_pipeline/train.py --config training_pipeline/configs/exp6_qwen3-0.6b.yaml
