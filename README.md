# approx-weighted-model-counting
Codes for the thesis presented for MSc Statistics with Data Science at University of Edinburgh, "*Scaling Approximate Weighted Model Counting*".

To reproduce the results, you need the *Cirkit* package which is not yet released. For markers, you can ask the School for this part of codes.

After you unzip the folder, you can place it at the root folder together with data_analysis, baselines, etc.

Then run `python main.py` at the root folder. Optional arguments include:

parser.add_argument('--debug', action='store_true', help='debug mode?')
- '--ds_class': Benchmark class, default='easy'.
- '--file_name': Name of the file, default='bayes_4step.cnf'.
- '--format': CNF file format, choices=['CAC', 'MIN', 'UNW', 'TRA'], default='MIN'.
- '--unweighted': Whether to run unweighted experiments, action='store_true'.
- '--model': Model choice, choices=['hmm', 'ind', 'pchmm', 'inh'], default='pchmm'.
- '--num_state': Hidden state count of HMM, default=64.
- '--reordered': Whether to use the reorder version HMM, action='store_true'.
- '--sample_size': Sampled data size, default=100000.
- '--batch_size': Size of batches, default=100.
- '--lr': Learning rate, default=0.1.

The results are automatically stored in the logs/ folder (have to create an empty logs/ folder first).

For baseline methods, simply cd into baselines/ and run `python` over the expected file. Optional arguments include:
- '--file_name': Name of the file, default='bayes_4step.cnf' (for both pysdd_wmc.py and pyapproxmc_wmc.py).
- '--unweighted': Whether to run unweighted experiments, action='store_true' (only for pysdd_wmc.py, weighted by default).
