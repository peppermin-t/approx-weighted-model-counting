#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=aWMC
#SBATCH --time=06:00:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=tc064-s2520995
#SBATCH -o /work/tc064/tc064/s2520995/approx-weighted-model-counting/errors/
#SBATCH -e /work/tc064/tc064/s2520995/approx-weighted-model-counting/errors/

module load pytorch/2.2.0-gpu
source ../conda-setup.sh
conda activate approxWMC

python main.py
