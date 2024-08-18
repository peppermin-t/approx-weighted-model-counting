import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

from utils import readCNF, construct_primal_graph



file_mode = "MIN"

ds_root = "../benchmarks/altogether"
ds_name = "easy"
ds_path = os.path.join(ds_root, ds_name)
print("Visualizing CNF as graphs from dataset class:", ds_path)
graph_path = os.path.join(ds_root, ds_name + "_primal_graphs")
print("Graphs saving to:", graph_path)

all_items = os.listdir(ds_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

treewidths = []

for fn in files:
	print(f"Processing {fn}:")

	with open(os.path.join(ds_path, fn)) as f:
		cnf, weights, _ = readCNF(f, mode=file_mode)
	cnf_set = [{abs(lit) for lit in clause} for clause in cnf]
	G = construct_primal_graph(cnf_set, len(weights))

	# Saving
	with open(os.path.join(graph_path, fn + ".pkl"), "wb") as f:
		pickle.dump(G, f)

	# Stats
	# print("Nodes: ", G.nodes())
	print("Nodes all recorded? ", len(G.nodes()) == len(weights))
	# print("Edges:", G.edges())
	
	# visualising the graph
	plt.figure(figsize=(8, 6))
	nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
	plt.title("Graph of CNF")
	plt.savefig(os.path.join(graph_path, fn + ".png"))
	plt.close()

	# tree width
	treewidth = nx.algorithms.approximation.treewidth_min_degree(G)[0]
	treewidths.append(treewidth)
	print("Tree width: ", treewidth)
