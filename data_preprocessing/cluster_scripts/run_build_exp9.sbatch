#!/bin/bash
#SBATCH --job-name=build_exp9
#SBATCH --output=build_exp9_data.log
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint="type_e"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

module load Python/PyTorch_GPU_v2.4

python data_preprocessing/build_exp9_en_only.py --output_dir ./exp9_en_only_dataset --seed 42
