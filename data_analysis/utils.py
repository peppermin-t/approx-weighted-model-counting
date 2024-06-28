import numpy as np


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

def evalCNF(cnf, x):  # 32.44s
    cnf_wrapped = [np.array(cls) for cls in cnf]

    cnfcnt, clscnt = x.shape[0], len(cnf)
    y = np.zeros((cnfcnt, clscnt))
    
    for i in range(cnfcnt):
        for j in range(clscnt):
            y[i, j] = np.any((x[i, abs(cnf_wrapped[j]) - 1] > 0) == (cnf_wrapped[j] > 0))
    
    return y
