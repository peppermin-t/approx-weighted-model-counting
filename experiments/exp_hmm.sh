#!/bin/bash

learning_rates=(0.1, 0.05)
num_states=(5, 10, 50)

target_directory="../benchmarks/altogether/easy"

if [ ! -d "$target_directory" ]; then
  echo "dir $target_directory does not exist"
  exit 1
fi

cd "$target_directory" || exit

for lr in "${learning_rates[@]}"; do
  for hs in "${num_states[@]}"; do
    for file in *; do
      if [ -f "$file" ]; then
	filename=$(basename "$file")
        echo "Running with lr=${lr}, num_hidden_state=${hs} on file ${filename}"
        python main.py --lr ${lr} --num_state ${hs} --filename ${filename}
      fi
    done
  done
done
