# Scaling Approximate Weighted Model Counting

Codes for the thesis presented for MSc Statistics with Data Science at University of Edinburgh, "*Scaling Approximate Weighted Model Counting*", supervised by Dr Antonio Vergari and Dr Emile van Krieken.

## Requirements

For main experiments:
- [Python](https://www.python.org/) - v3.11.5
- [Pytorch](https://pytorch.org/) - v2.2.0
- [Weight&Biases](https://wandb.ai/) - v0.16.3

For baselines:
- [PySDD](https://github.com/wannesm/PySDD) - v0.2.12
- [PyapproxMC](https://github.com/meelgroup/approxmc) - v4.1.24

## Running

To reproduce the results, you need the *Cirkit* package by the [APRIL](https://april-tools.github.io/) lab which is not yet released. For markers, you can ask the School for this part of codes, and place it at the root folder paralleled with `data_analysis/`, `baselines/`, etc. after you unzip the folder.

Benchmarks can be found at [benchmarks.zip](https://github.com/vardigroup/ADDMC/releases/download/v1.0.0/benchmarks.zip), store at root before running.

Then run `python main.py` at the root folder. Optional arguments include:
- `--debug`: Whether to run in debug mode, action='store_true'.
- `--ds_class`: Benchmark class, default='easy'.
- `--file_name`: Name of the file, default='bayes_4step.cnf'.
- `--format`: CNF file format, choices=['CAC', 'MIN', 'UNW', 'TRA'], default='MIN'.
- `--unweighted`: Whether to run unweighted experiments, action='store_true'.
- `--model`: Model choice, choices=['hmm', 'ind', 'pchmm', 'inh'], default='pchmm'.
- `--num_state`: Hidden state count of HMM, default=64.
- `--reordered`: Whether to use the reorder version HMM, action='store_true'.
- `--sample_size`: Sampled data size, default=100000.
- `--batch_size`: Size of batches, default=100.
- `--lr`: Learning rate, default=0.1.

The results are automatically stored in the `logs/` folder (create an empty folder at root first). Also need to create an empty `models/` folder beforehand to store models.

For baseline methods, simply cd into baselines/ and run `python` over the expected file. Optional arguments include:
- `--file_name`: Name of the file, default='bayes_4step.cnf' (for both pysdd_wmc.py and pyapproxmc_wmc.py).
- `--unweighted`: Whether to run unweighted experiments, action='store_true' (only for pysdd_wmc.py, weighted by default).

For example:
```bash
$ python main.py --file_name bayes_4step.cnf --num_state 128 --reordered
```
The above line runs an experiment on CNF file `bayes_4step.cnf` in the easy class, based on a reordered version of hidden Markov model circuit (HMMC) of 128 hidden state counts and sampling size of 100,000, batch size of 100, and learning rate of 0.1.
