import math
import numpy as np
import time
import os
import random
import argparse
import pyapproxmc

from data_analysis.utils import readCNF

def compute_pyapproxmc(cnf, print_time=True):
    t0 = time.time()

    c = pyapproxmc.Counter()
    for clause in cnf:
        c.add_clause(clause)
    count = c.count()

    if print_time:
        t1 = time.time()
        print(f"Pyapproxmc spending time: {t1 - t0}")

    return count


if __name__ == "__main__":
    
    # seed
    random.seed(42)
    np.random.seed(42)
    ds_root = "benchmarks/altogether"
    ds_class = "easy"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='bayes_4step.cnf', type=str, help='Name of the file')
    config = vars(parser.parse_args())
    
    cnf_path = os.path.join(ds_root, ds_class, config['file_name'])

    with open(cnf_path) as f:
        cnf, w, _ = readCNF(f, mode="MIN")
        
    print(f"Calculating approxmc result of {config['file_name']}...")
    res = compute_pyapproxmc(cnf)
    a, b = res
    
    # normalised
    nvar = len(w)
    print(f"The log approx MC result is: {math.log(a) + (b - nvar) * math.log(2)}")
    print(f"The approx MC result is: {a * math.exp2(b - nvar)}")
