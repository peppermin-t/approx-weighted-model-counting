import torch
import numpy as np

def readCNF(f, mode="CAC"):
    in_data =  [line.strip() for line in f if line.strip()]
    cnf_ptr = 0

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
            if mode == "CAC":
                ind = int(tokens[1]) - 1
                w = float(tokens[2])
                if w == -1:
                    weights[ind, ] = [1, 1]
                else:
                    weights[ind, ] = [w, 1 - w]
            else:  # "MIN"
                for i in range(varcnt):
                    weights[i, 0] = tokens[2 * i]
                    weights[i, 1] = tokens[2 * i + 1]

        elif len(tokens) != 0 and tokens[0] not in ("c", "%"):
            for tok in tokens:
                lit = int(tok)
                if lit == 0:
                    cnf_ptr += 1
                else:
                    cnf[cnf_ptr].append(lit)

    return cnf, weights

def evalCNF(cnf, x):
    y = np.zeros(len(cnf))
    for i in range(len(cnf)):
        for lit in cnf[i]:
            if (x[abs(lit) - 1] == 1 and lit > 0) or (x[abs(lit) - 1] == 0 and lit < 0):
                y[i] = 1
                break
    
    return torch.from_numpy(y)
