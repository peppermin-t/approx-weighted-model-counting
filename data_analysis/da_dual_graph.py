import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from utils import readCNF

def construct_graph(cnf_set, nvar):
	G = nx.Graph()
	G.add_nodes_from(list(range(nvar)))
	for clause in cnf_set:
		for i in range(len(clause)):
			for j in range(i + 1, len(clause)):
				G.add_edge(clause[i] - 1, clause[j] - 1)

	return G

file_mode = "MIN"

ds_root = "../benchmarks/altogether"
ds_name = "easy"
ds_path = os.path.join(ds_root, ds_name)
print("Visualizing CNF as graphs from dataset class:", ds_path)
graph_path = os.path.join(ds_root, ds_name + "_dual_graphs")
print("Graphs saving to:", graph_path)

all_items = os.listdir(ds_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

for fn in files:
	print(f"Processing {fn}:")

	with open(os.path.join(ds_path, fn)) as f:
		cnf, weights, _ = readCNF(f, mode=file_mode)
	cnf_set = [{abs(lit) for lit in clause} for clause in cnf]
	G = construct_graph(cnf_set, len(weights))

	# Saving
	with open("graph.pkl", "wb") as f:
		pickle.dump(G, f)

	# Stats
	print("Nodes: ", G.nodes())
	print("Nodes all recorded? ", len(G.nodes()) == len(weights))
	print("Edges:", G.edges())
	
	# visualising the graph
	plt.figure(figsize=(8, 6))
	nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
	plt.title("Graph of CNF")
	plt.savefig(os.path.join(graph_path, fn + ".png"))
	plt.close()

	# tree width
	print("Tree width: ", nx.algorithms.approximation.treewidth_min_degree(G))
