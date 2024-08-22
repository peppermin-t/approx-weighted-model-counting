import math
import numpy as np
import time
import os
import random
import argparse
from pysdd.sdd import SddManager

from data_analysis.utils import readCNF

def compute_pysdd(fstr, weights, print_time=True):
    t0 = time.time()
    
    print("Start convertion from cnf to sdd..")
    manager, node = SddManager.from_cnf_file(bytes(fstr, encoding='utf8'))
    print("End convertion.")

    wmc = node.wmc(log_mode=True)
    if weights is not None:
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
        log_res = log_res - nvar * math.log(2)
    print(f"The log approx MC result is: {log_res}")
    print(f"The approx MC result is: {math.exp(log_res)}")


