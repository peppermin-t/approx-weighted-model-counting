import argparse

def parsearg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dsclass', default='easy', type=str, help='Benchmark class')
    parser.add_argument('--filename', default='pseudoweighted_emptyroom_d4_g2_p_t2.cnf', type=str, help='Name of the file')
    parser.add_argument('--modelpth', default='models/easy', type=str, help='Path of models')

    parser.add_argument('--format', type=str, choices=['CAC', 'MIN', 'UNW', 'TRA'], default='MIN', help='CNF file format')
    parser.add_argument('--model', type=str, choices=['hmm', 'ind'], default='hmm', help='Model choice')
    parser.add_argument('--num_state', type=int, default=10, help='Hidden state count of HMM')
    parser.add_argument('--sample_size', type=int, default=100000, help='Sampled data size')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of batches')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    return args
