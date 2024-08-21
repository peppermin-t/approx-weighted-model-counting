from pysdd.sdd import SddManager
import math
import numpy as np
import time
from tqdm import tqdm
import os
import json
import random
import argparse

from data_analysis.utils import readCNF

def compute_pysdd(fstr, weights, print_time=True):
    t0 = time.time()
    
    print("Start convertion from cnf to sdd..")
    manager, node = SddManager.from_cnf_file(bytes(fstr, encoding='utf8'))
    print("End convertion.")

    wmc = node.wmc(log_mode=True)

    # (Weighted) Model Counting
    if weights:
        formatted_w = np.concatenate((weights[::-1, 1], weights[:, 0]))
        manager.set_prevent_transformation(prevent=False)
        wmc.set_literal_weights_from_array(np.log(formatted_w))
    print("Start propagating...")
    w = wmc.propagate()
    print("End propogating.")

    if print_time:
        t1 = time.time()
        print(f"Pysdd spending time: {t1 - t0}")

    return w


if __name__ == "__main__":

    # ds_root = "../benchmarks/altogether"
    # ds_name = "pseudoweighted"
    # clscnt_thr = 150
    # varcnt_thr = 791

    # dataset_path = os.path.join(ds_root, ds_name + "_MINIC2D")
    # answer_path = os.path.join(ds_root, ds_name + '_logans.json')
    # print("Selecting benchmarks from:", dataset_path)
    # print("Exact WMC answers' output path:", answer_path)

    # all_items = os.listdir(dataset_path)
    # files = [fn for fn in all_items if os.path.isfile(os.path.join(dataset_path, fn))]

    # for fn in tqdm(files):
    #     with open(answer_path) as ans:
    #         answers = json.load(ans)
    #     if not fn in answers:
    #         fstr = os.path.join(dataset_path, fn)
    #         with open(fstr) as f:
    #             cnf, weights, _ = readCNF(f, mode="MIN")
    #         if len(cnf) >= clscnt_thr or len(weights) >= varcnt_thr: continue
    #         print(f"Start processing {fn}:")
    #         probs = weights / weights.sum(axis=1, keepdims=True)
    #         res = compute_exact_WMC_pysdd_from_file(fstr, probs, log=True)
    #         answers[fn] = res

    #         with open(answer_path, "w") as ans:
    #             json.dump(answers, ans, indent=4)
                
    
    # seed
    random.seed(42)
    np.random.seed(42)
    ds_root = "benchmarks/altogether"
    ds_class = "easy"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='bayes_4step.cnf', type=str, help='Name of the file')
    parser.add_argument('--unweighted', action='store_true', help='unweighted?')
    config = vars(parser.parse_args())
    
    cnf_path = os.path.join(ds_root, ds_class, config['file_name'])

    with open(cnf_path) as f:
        cnf, w, _ = readCNF(f, mode="MIN")
    w = w / w.sum(axis=1, keepdims=True)

    print(f"Calculating exact mc result of {config['file_name']}...")
    log_res = compute_pysdd(cnf_path, weights=None if config['unweighted'] else w)

    if config['unweighted']:
        nvar = len(w)
        print(f"The log approx MC result is: {log_res - nvar * math.log(2)}")
        print(f"The approx MC result is: {math.exp(log_res - nvar * math.log(2))}")
    else:
        print(f"The log approx MC result is: {log_res}")
        print(f"The approx MC result is: {math.exp(log_res)}")


