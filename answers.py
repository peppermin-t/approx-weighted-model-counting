from itertools import product


def compute_w(weights, perm):
    res = 1
    for v in range(len(perm)):
        if perm[v] == 0: res *= weights[v, 1]
        else: res *= weights[v, 0]

    return res

def compute_WMC(weights, cnf):
    weights = weights / weights.sum(axis=1).reshape((-1, 1))  # normalise
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
