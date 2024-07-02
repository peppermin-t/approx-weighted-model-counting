#!/bin/bash

module load anaconda/
conda activate approxW

learning_rates=(0.05 0.1)
num_states=(10 25)

target_directory="benchmarks/altogether/easy"

if [ ! -d "$target_directory" ]; then
  echo "dir $target_directory does not exist"
  exit 1
fi

for file in "$target_directory"/*; do
  for lr in "${learning_rates[@]}"; do
    for hs in "${num_states[@]}"; do
      if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Running with lr=${lr}, num_hidden_state=${hs} on file ${filename}"
        python main.py --lr ${lr} --num_state ${hs} --filename ${filename}
      fi
    done
  done
done
