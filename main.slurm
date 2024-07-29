#!/bin/bash

#SBATCH --job-name=aWMC
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc064-s2520995
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH -o ./errors/aWMC_%j.out

module load pytorch/2.2.0-gpu

source /work/tc064/tc064/s2520995/myenv/bin/activate

python main.py --wandb_deac --num_state 128
