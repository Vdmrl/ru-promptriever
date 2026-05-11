#!/bin/bash
#SBATCH --job-name=build_exp8
#SBATCH --output=build_exp8_data.log
#SBATCH --partition=cpu-e-quick
#SBATCH --constraint="type_e"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load Python/PyTorch_GPU_v2.4

python data_preprocessing/build_continue_training_dataset_with_eng.py \
    --output_dir ./exp8_gold_mix_dataset \
    --seed 42
