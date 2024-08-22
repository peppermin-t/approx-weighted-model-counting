import argparse

def parsearg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true', help='debug mode?')

    parser.add_argument('--ds_class', default='easy', type=str, help='Benchmark class')
    parser.add_argument('--file_name', default='bayes_4step.cnf', type=str, help='Name of the file')

    parser.add_argument('--format', type=str, choices=['CAC', 'MIN', 'UNW', 'TRA'], default='MIN', help='CNF file format')
    parser.add_argument('--model', type=str, choices=['hmm', 'ind', 'pchmm', 'inh'], default='pchmm', help='Model choice')
    parser.add_argument('--num_state', type=int, default=64, help='Hidden state count of HMM')
    parser.add_argument('--reordered', action='store_true', help='reorder?')
    parser.add_argument('--unweighted', action='store_true', help='unweighted?')
    
    parser.add_argument('--sample_size', type=int, default=100000, help='Sampled data size')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of batches')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    return args
