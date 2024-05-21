import numpy as np

def readCNF(f):  # bayes
    in_data =  f.readlines()
    cnf = [[]]

    for line in in_data:
        tokens = line.split()
        if tokens[0] == "p":
            weights = np.full(int(tokens[2]), 0.5)
        elif tokens[0] == "w":
            weights[int(tokens[1]) - 1] = float(tokens[2])  # -1
        elif len(tokens) != 0 and tokens[0] not in ("c", "%"):
            for tok in tokens:
                lit = int(tok)
                if lit == 0:
                    cnf.append(list())
                else:
                    cnf[-1].append(lit)

    assert len(cnf[-1]) == 0
    cnf.pop()

    return np.array(cnf), weights

def evalCNF(cnf, x):
    y = np.zeros(len(cnf))
    for i in range(len(cnf)):
        for lit in cnf[i, ]:
            if (x[abs(lit) - 1] == 1 and lit > 0) or (x[abs(lit) - 1] == 0 and lit < 0):
                y[i] = 1
                break
    
    return y