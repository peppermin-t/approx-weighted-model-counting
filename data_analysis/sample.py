import numpy as np
import os
import time

from utils import readCNF, evalCNF

# def sample_y(probs, cnf, size):  # consistently on torch
# 	dist_x = Bernoulli(torch.from_numpy(probs))
# 	x = dist_x.sample(torch.tensor([size]))
# 	return torch.from_numpy(evalCNF(cnf, x.numpy()))

def sample_y(probs, cnf, size):  # faster on cpu
    x = np.random.binomial(1, probs, (size, len(probs)))
    return evalCNF(cnf, x)
    # return torch.from_numpy(evalCNF(cnf, x))


sample_size = 100000
file_mode = "MIN"

# torch.manual_seed(0)
np.random.seed(0)

ds_root = "../benchmarks/altogether"
ds_name = "easy"
ds_path = os.path.join(ds_root, ds_name)
print("Sampling from dataset class:", ds_path)
smp_path = os.path.join(ds_root, ds_name + "_samples")
print("Samples saving to:", smp_path)

all_items = os.listdir(ds_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

for fn in files:
	with open(os.path.join(ds_path, fn)) as f:
		cnf, weights, _ = readCNF(f, mode=file_mode)
		
	probs = (weights / weights.sum(axis=1, keepdims=True))[:, 0]
	
	t0 = time.time()
	y = sample_y(probs, cnf, size=sample_size)
	t1 = time.time()
	print(f"File {fn} sampling time: {t1 - t0:.2f}")
	
	np.save(open(os.path.join(smp_path, fn + ".npy"), "wb"), y)
