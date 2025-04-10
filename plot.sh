#!/bin/bash
#SBATCH --job-name=plot_test
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:10:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1


source activate eval_24

python plot_results.py
