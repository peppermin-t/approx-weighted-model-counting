#!/bin/bash

#SBATCH --job-name=aWMC
#SBATCH --time=20:00:00
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc064-s2520995
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH -o ./errors/aWMC_%j.out

module load pytorch/2.2.0-gpu

source /work/tc064/tc064/s2520995/myenv/bin/activate
export WANDB_MODE=offline

# 3
# python main.py --file_name pseudoweighted_flip_1_p_t1.cnf --num_state 128 --unweighted

# 7
# python main.py --file_name pseudoweighted_bomb_b5_t1_p_t1.cnf --num_state 128 --reordered --unweighted

# 11
# python main.py --file_name bayes_4step.cnf --num_state 128 --reordered --unweighted

# 19
# python main.py --file_name bayes_90-12-3-q.cnf --num_state 128 --unweighted
# python main.py --file_name bayes_90-12-3-q.cnf --num_state 128 --reordered --unweighted

# 19
python main.py --file_name bayes_50-12-2-q.cnf --num_state 128 --unweighted
python main.py --file_name bayes_50-12-2-q.cnf --num_state 128 --reordered --unweighted

