#!/bin/bash

learning_rates=(0.1 0.05 0.01)
num_states=(5 10 50)

for lr in "${learning_rates[@]}"; do
  for hs in "${num_states[@]}"; do
    echo "Running with lr=${lr} and hidden_state=${hs}"
    python main.py --lr ${lr} --num_state ${hs}
  done
done
