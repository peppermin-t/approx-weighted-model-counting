import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from utils import readCNF

def construct_graph(cnf_set):
	n = len(cnf_set)
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	for i in range(n):
		for j in range(i + 1, n):
			if cnf_set[i] & cnf_set[j]:
				G.add_edge(i, j)

	return G

file_mode = "MIN"

ds_root = "../benchmarks/altogether"
ds_name = "easy"
ds_path = os.path.join(ds_root, ds_name)
print("Visualizing CNF as graphs from dataset class:", ds_path)
graph_path = os.path.join(ds_root, ds_name + "_primal_graphs")
print("Graphs saving to:", graph_path)

all_items = os.listdir(ds_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

for fn in files:
	print(f"Processing {fn}:")

	with open(os.path.join(ds_path, fn)) as f:
		cnf, weights, _ = readCNF(f, mode=file_mode)
	cnf_set = [{abs(lit) for lit in clause} for clause in cnf]
	G = construct_graph(cnf_set)
	
	# Saving
	with open(os.path.join(graph_path, fn + ".pkl"), "wb") as f:
		pickle.dump(G, f)

	# Stats
	print("Nodes: ", G.nodes())
	print("Nodes all recorded? ", len(G.nodes()) == len(cnf))
	print("Edges:", G.edges())
	
	# visualising the graph
	pos = nx.spring_layout(G)
	# pos = nx.kamada_kawai_layout(G)
	plt.figure(figsize=(16, 12))
	nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=12, edge_color='gray')
	plt.title("Graph of CNF", fontsize=16)
	plt.savefig(os.path.join(graph_path, fn + ".png"))
	plt.close()

	# tree width
	print("Tree width: ", nx.algorithms.approximation.treewidth_min_degree(G))
