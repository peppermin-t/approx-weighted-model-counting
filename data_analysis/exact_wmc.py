from pysdd.sdd import SddManager, Vtree
import math
import numpy as np
import time
from tqdm import tqdm
import os
import json
from itertools import product

from utils import readCNF

def compute_exact_WMC_pysdd_from_file(fstr, weights, weighted=True, log=False, print_time=True):  # fast, but limited to MINIC2D format
    if print_time: t0 = time.time()
    print("Start convertion from cnf to sdd..")
    manager, node = SddManager.from_cnf_file(bytes(fstr, encoding='utf8'))
    print("End convertion.")

    wmc = node.wmc(log_mode=True)

    # (Weighted) Model Counting
    if weighted:
        formatted_w = np.concatenate((weights[::-1, 1], weights[:, 0]))
        manager.set_prevent_transformation(prevent=False)
        wmc.set_literal_weights_from_array(np.log(formatted_w))
    print("Start propagating...")
    w = wmc.propagate()
    print("End propogating.")

    if print_time:
        t1 = time.time()
        print(f"Pysdd spending time: {t1 - t0}")
    
    if log: res_w = w
    else: res_w = math.exp(w)

    return res_w


def compute_exact_WMC_pysdd_from_list(cnf, weights, weighted=True):  # valid, but too slow in the list-to-sdd phase
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

def compute_w(weights, perm):
    res = 1
    for v in range(len(perm)):
        if perm[v] == 0: res *= weights[v, 1]
        else: res *= weights[v, 0]

    return res

def compute_exact_WMC_raw(weights, cnf):
    # weights = weights / weights.sum(axis=1).reshape((-1, 1))  # normalise
    varcnt = len(weights)

    binary_permutations = list(product([0, 1], repeat=varcnt))

    res = 0
    for perm in binary_permutations:
        flag_1 = True
        for cls in cnf:
            flag_2 = False
            for lit in cls:
                if lit < 0 and perm[-lit - 1] == 0 or lit > 0 and perm[lit - 1] == 1:
                    flag_2 = True
                    break
            if not flag_2:
                flag_1 = False
                break
        if flag_1: res += compute_w(weights, perm)

    return res


if __name__ == "__main__":

    ds_root = "../benchmarks/altogether"
    ds_name = "pseudoweighted"
    clscnt_thr = 150
    varcnt_thr = 791

    dataset_path = os.path.join(ds_root, ds_name + "_MINIC2D")
    answer_path = os.path.join(ds_root, ds_name + '_logans.json')
    print("Selecting benchmarks from:", dataset_path)
    print("Exact WMC answers' output path:", answer_path)

    all_items = os.listdir(dataset_path)
    files = [fn for fn in all_items if os.path.isfile(os.path.join(dataset_path, fn))]

    for fn in tqdm(files):
        with open(answer_path) as ans:
            answers = json.load(ans)
        if not fn in answers:
            fstr = os.path.join(dataset_path, fn)
            with open(fstr) as f:
                cnf, weights, _ = readCNF(f, mode="MIN")
            if len(cnf) >= clscnt_thr or len(weights) >= varcnt_thr: continue
            print(f"Start processing {fn}:")
            probs = weights / weights.sum(axis=1, keepdims=True)
            res = compute_exact_WMC_pysdd_from_file(fstr, probs, log=True)
            answers[fn] = res

            with open(answer_path, "w") as ans:
                json.dump(answers, ans, indent=4)

