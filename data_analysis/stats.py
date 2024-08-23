from utils import readCNF
import os
import matplotlib.pyplot as plt
import numpy as np

# Calculating statitics of dataset classes

def calc_stats(dsclass, mode):
	stats = []
	
	for rt, _, fns in os.walk(os.path.join("../benchmarks/altogether", dsclass)):
		for fn in fns:
			with open(os.path.join(rt, fn)) as f:
				cnf, weights, max_clslen = readCNF(f, mode)
			stats.append([len(cnf), len(weights), max_clslen])
	return np.array(stats)

# excluding examples (only for demonstration or testing purposes)
# two type of dataset: bayes & pseudoweighted
# no suffix and _MINIC2D means CACHET and MINIC2D format of the same dataset
# _w means the corresponding weight-only files

# "easy" is part of the combined dataset, with tolerable pysdd solving time
#	(excluding log normalised WMC result of -Infinity)
# "hard" is part of the pseudoweighted dataset
stats1 = calc_stats("pseudoweighted_MINIC2D", "MIN")
stats2 = calc_stats("bayes_MINIC2D", "MIN")

stats3 = calc_stats("easy", "MIN")
stats4 = calc_stats("hard", "MIN")

print(stats1.mean(axis=0))
print(stats2.mean(axis=0))
print(stats3.mean(axis=0))
print(stats4.mean(axis=0))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(stats1[:, 0], label='pseudoweighted', marker='o')
axs[0].plot(stats2[:, 0], label='bayes', marker='s')
axs[0].plot(stats3[:, 0], label='easy', marker='^')
axs[0].plot(stats4[:, 0], label='hard', marker='v')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Clause Counts')
axs[0].set_title('Clause Count Comparison')
axs[0].legend()
axs[0].grid(True)

# Plot varcnts
axs[1].plot(stats1[:, 1], label='pseudoweighted', marker='o')
axs[1].plot(stats2[:, 1], label='bayes', marker='s')
axs[1].plot(stats3[:, 1], label='easy', marker='^')
axs[1].plot(stats4[:, 1], label='hard', marker='v')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Var Counts')
axs[1].set_title('Variable Count Comparison')
axs[1].legend()
axs[1].grid(True)

# Plot max_clslens
axs[2].plot(stats1[:, 2], label='pseudoweighted', marker='o')
axs[2].plot(stats2[:, 2], label='bayes', marker='s')
axs[2].plot(stats3[:, 2], label='easy', marker='^')
axs[2].plot(stats4[:, 2], label='hard', marker='v')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Max Clause Lengths')
axs[2].set_title('Max Clause Length Comparison')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('da.png')
