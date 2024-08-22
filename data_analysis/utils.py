import math
import time
import numpy as np
import networkx as nx
import torch
from torch.distributions.bernoulli import Bernoulli


def construct_dual_graph(cnf_set):
	n = len(cnf_set)
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	for i in range(n):
		for j in range(i + 1, n):
			if cnf_set[i] & cnf_set[j]:
				G.add_edge(i, j)

	return G

def construct_primal_graph(cnf_set, nvar):
	G = nx.Graph()
	G.add_nodes_from(list(range(nvar)))
	for clause in cnf_set:
		clause = list(clause)
		for i in range(len(clause)):
			for j in range(i + 1, len(clause)):
				G.add_edge(clause[i] - 1, clause[j] - 1)

	return G

def dfs_all_components(G):
    dfs_order = []
    
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        start_node = next(iter(component))
        dfs_order.extend(nx.dfs_preorder_nodes(subgraph, start_node))
    
    return dfs_order

def calc_error(a, b, exp=False):
    return abs(math.exp(a) - math.exp(b)) if exp else abs(a - b)

def sample_y(probs, cnf, size, device):  # consistently on torch
	dist_x = Bernoulli(torch.from_numpy(probs).to(device))
	x = dist_x.sample(torch.tensor([size]))
	return evalCNF(cnf, x.cpu().numpy())

# def sample_y(probs, cnf, size):  # faster on cpu
#     x = np.random.binomial(1, probs, (size, len(probs)))
#     return evalCNF(cnf, x)


def sample(cnf, weights, sample_size, device, unweighted=False):
    probs = (weights / weights.sum(axis=1, keepdims=True))[:, 0]
    if unweighted:
        probs = np.full(probs.shape, 0.5)
    t0 = time.time()
    y = sample_y(probs, cnf, size=sample_size, device=device)
    t1 = time.time()
    print(f"Sampling time: {t1 - t0:.2f}")
    return y

def readCNF(f, mode="MIN"):
    in_data =  [line.strip() for line in f if line.strip()]
    cnf_ptr = 0
    max_clslen = 0

    for line in in_data:
        tokens = line.split()
        if tokens[0] == "p":
            varcnt = int(tokens[2])
            clscnt = int(tokens[3])
            if mode == "UNW":
                weights = np.full((varcnt, 2), 1.0)
            else:
                weights = np.full((varcnt, 2), 0.5)
            cnf = [[] for _ in range(int(clscnt))]
        elif tokens[0] == "w":
            w = float(tokens[2])
            if mode == "CAC":
                ind = int(tokens[1]) - 1
                if w == -1:
                    weights[ind, ] = [1, 1]
                else:
                    weights[ind, ] = [w, 1 - w]
            else:  # track
                ind = abs(int(tokens[1])) - 1
                weights[ind, int(int(tokens[1]) < 0)] = w
        elif mode == "MIN" and tokens[0] == "c" and tokens[1] == "weights":
            for i in range(varcnt):
                weights[i, 0] = tokens[2 * i + 2]
                weights[i, 1] = tokens[2 * i + 3]
        elif len(tokens) != 0 and tokens[0] not in ("c", "%"):
            if len(tokens) > max_clslen: max_clslen = len(tokens)
            for tok in tokens:
                lit = int(tok)
                if lit == 0:
                    cnf_ptr += 1
                else:
                    cnf[cnf_ptr].append(lit)

    return cnf, weights, max_clslen

def evalCNF(cnf, x):
    cnf_wrapped = [np.array(cls) for cls in cnf]

    cnfcnt, clscnt = x.shape[0], len(cnf)
    y = np.zeros((cnfcnt, clscnt))
    
    for i in range(cnfcnt):
        for j in range(clscnt):
            y[i, j] = np.any((x[i, abs(cnf_wrapped[j]) - 1] > 0) == (cnf_wrapped[j] > 0))
    
    return y
