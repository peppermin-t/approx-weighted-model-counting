from utils import readCNF
import os
import matplotlib.pyplot as plt

def calc_stats(dsclass, mode):
	clscnts, varcnts, max_clslens = [], [], []
	
	for rt, _, fns in os.walk(os.path.join("benchmarks/altogether", dsclass)):
		for fn in fns:
			with open(os.path.join(rt, fn)) as f:
				cnf, weights, max_clslen = readCNF(f, mode)
			clscnts.append(len(cnf))
			varcnts.append(len(weights))
			max_clslens.append(max_clslen)
	return clscnts, varcnts, max_clslens

clscnts, varcnts, max_clslens = calc_stats("pseudoweighted_MINIC2D", "MIN")
plt.figure()
plt.plot(clscnts)
plt.plot(varcnts)
plt.plot(max_clslens)
plt.show()
