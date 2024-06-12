import numpy as np
import argparse

def parsearg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', default='benchmarks/altogether/easy/pseudoweighted_bomb_b5_t1_p_t1.cnf', type=str, help='Name of the file')
    parser.add_argument('--modelpth', default='models/easy', type=str, help='Path of models')
    parser.add_argument('--format', type=str, choices=['CAC', 'MIN', 'UNW', 'TRA'], default='MIN', help='CNF file format')
    parser.add_argument('--model', type=str, choices=['hmm', 'ind'], default='hmm', help='Model choice')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sampled data size')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of batches')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()
    return args

def readCNF(f, mode="CAC"):
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

def evalCNF(cnf, x):  # possible optimisation?
    y = np.zeros((x.shape[0], len(cnf)))

    for i in range(x.shape[0]):
        for j in range(len(cnf)):
            for lit in cnf[j]:
                if not ((x[i, abs(lit) - 1] > 0) ^ (lit > 0)):
                    y[i, j] = 1
                    break
    
    return y
