#!/bin/sh
# Grid Engine options (lines prefixed with #$)
# job name
#$ -N da_primal_graph
# use the current working dir              
#$ -wd /exports/eddie/scratch/s2520995/approx-weighted-model-counting/data_analysis/

#$ -l rl9=true

# runtime limit of 1 hour
#$ -l h_rt=02:00:00

# Request 5 GB system RAM available to the job is the value specified here multiplied by the number of requested GPU (above)
#$ -l h_vmem=32G

# email
#$ -M chenyinjia2000@gmail.com
#$ -m beas
#$ -o /exports/eddie/scratch/s2520995/approx-weighted-model-counting/logs/
#$ -e /exports/eddie/scratch/s2520995/approx-weighted-model-counting/errors/

. /etc/profile.d/modules.sh
module load anaconda/
conda activate approxW
# Run my script
python da_primal_graph.py
