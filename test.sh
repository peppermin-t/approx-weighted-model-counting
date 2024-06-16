#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# job name
#$ -N test

# use the current working dir              
#$ -wd /exports/eddie/scratch/s2520995/approx-weighted-model-counting                

# Request one GPU in the gpu queue:
# -q gpu
# -pe gpu-a100 1

# runtime limit of 15 minutes
#$ -l h_rt=00:01:00

# Request 5 GB system RAM available to the job is the value specified here multiplied by
#  the number of requested GPU (above)
#$ -l h_vmem=1G

# email
#$ -M [chenyinjia2000@gmail.com](mailto:chenyinjia2000@gmail.com)<[https://mailto:chenyinjia2000@gmail.com](https://mailto:chenyinjia2000@gmail.com/)>
#$ -m beas
#$ -o /exports/eddie/scratch/s2520995/approx-weighted-model-counting/logs/test.o

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda
module load anaconda/
# Activate conda env
conda activate approxWMC
# Run my script
python test.py
