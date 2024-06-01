from pysdd.sdd import SddManager, Vtree
import math
import numpy as np

import time
from tqdm import tqdm
from utils import readCNF

def compute_exact_WMC_from_file(fstr, weights, weighted=True):  # fast, but limited to MINIC2D format
    t0 = time.time()
    manager, node = SddManager.from_cnf_file(bytes(fstr, encoding='utf8'))  # sdd.count(), sdd.index()

    wmc = node.wmc(log_mode=True)

    # (Weighted) Model Counting
    if weighted:
        formatted_w = np.concatenate((weights[::-1, 1], weights[:, 0]))
        manager.set_prevent_transformation(prevent=False)
        wmc.set_literal_weights_from_array(np.log(formatted_w))
    w = wmc.propagate()
    res_w = math.exp(w)

    t1 = time.time()
    print(f"Pysdd spending time: {t1 - t0}")
    return res_w


def compute_exact_WMC(cnf, weights, weighted=True):  # valid, but too slow in the list-to-sdd phase
    varcnt = weights.shape[0]

    vtree = Vtree(var_count=varcnt, vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)

    sdd = manager.true()

    t0 = time.time()
    for clause in tqdm(cnf):
        clause_sdd = manager.false()
        for literal in clause:
            var = abs(literal)
            lit_sdd = manager.literal(var if literal > 0 else -var)
            clause_sdd = clause_sdd | lit_sdd
        sdd = sdd & clause_sdd
    
    t1 = time.time()
    print(f"List to SDD time: {t1 - t0}")
    wmc = sdd.wmc(log_mode=True)

    # (Weighted) Model Counting
    if weighted:
        formatted_w = np.concatenate((weights[::-1, 1], weights[:, 0]))
        wmc.set_literal_weights_from_array(np.log(formatted_w))
    w = wmc.propagate()
    res_w = math.exp(w)
    t2 = time.time()
    print(f"Propogate time: {t2 - t1}")
    return res_w

# tesing 
if __name__ == "__main__":
    # this method only suits MINIC2D format CNF file
    fstr_MIN = "benchmarks/altogether/bayes_MINIC2D/50-12-3-q.cnf"
    with open(fstr_MIN) as f:
        cnf, weights, _ = readCNF(f, mode="MIN")

    res = compute_exact_WMC_from_file(fstr_MIN, weights)
    print(res)

    res = compute_exact_WMC_from_file(fstr_MIN, weights, weighted=False)
    print(res)
