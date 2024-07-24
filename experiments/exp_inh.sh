#!/bin/sh
# Grid Engine options (lines prefixed with #$)
# job name
#$ -N exp_inh
# use the current working dir              
#$ -wd /exports/eddie/scratch/s2520995/approx-weighted-model-counting

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l rl9=true

# runtime limit of 1 hour
#$ -l h_rt=24:00:00

# Request 32 GB system RAM available to the job is the value specified here multiplied by the number of requested GPU (above)
#$ -l h_vmem=32G

# email
#$ -M chenyinjia2000@gmail.com
#$ -m beas
#$ -o /exports/eddie/scratch/s2520995/approx-weighted-model-counting/errors/
#$ -e /exports/eddie/scratch/s2520995/approx-weighted-model-counting/errors/

# Initialise the environment modules and load CUDA version 12.1.1
. /etc/profile.d/modules.sh
module load cuda/12.1.1
module load anaconda/
# Activate conda env
conda activate approxW

num_states=(25)

target_directory="benchmarks/altogether/easy"

if [ ! -d "$target_directory" ]; then
  echo "dir $target_directory does not exist"
  exit 1
fi

for file in "$target_directory"/*; do
  for hs in "${num_states[@]}"; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      echo "Running with lr=${lr}, num_hidden_state=${hs} on file ${filename}"
      python main.py --model inh --num_state ${hs} --filename ${filename}
    fi
  done
done
