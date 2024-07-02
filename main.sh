#!/bin/sh
# Grid Engine options (lines prefixed with #$)
# job name
#$ -N aWMC
# use the current working dir              
#$ -wd /exports/eddie/scratch/s2520995/approx-weighted-model-counting

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l rl9=true

# runtime limit of 1 hour
#$ -l h_rt=00:15:00

# Request 5 GB system RAM available to the job is the value specified here multiplied by the number of requested GPU (above)
#$ -l h_vmem=32G

# email
#$ -M chenyinjia2000@gmail.com
#$ -m beas
#$ -o /exports/eddie/scratch/s2520995/approx-weighted-model-counting/logs/
#$ -e /exports/eddie/scratch/s2520995/approx-weighted-model-counting/errors/

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda/12.1.1
module load anaconda/
# Activate conda env
conda activate approxW
# Run my script
python main.py
