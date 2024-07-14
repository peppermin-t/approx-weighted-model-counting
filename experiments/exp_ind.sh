#!/bin/bash

module load anaconda/
conda activate approxW

learning_rates=(0.5 0.05)

target_directory="benchmarks/altogether/easy"

if [ ! -d "$target_directory" ]; then
  echo "dir $target_directory does not exist"
  exit 1
fi

for lr in "${learning_rates[@]}"; do
  for file in "$target_directory"/*; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      echo "Running with lr=${lr}, on file ${filename}"
      python main.py --model "ind" --lr ${lr} --filename ${filename}
    fi
  done
done

