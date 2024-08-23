import os
import networkx as nx
import matplotlib.pyplot as plt

from utils import readCNF, construct_primal_graph

# ploting the primal graphs of the formulas in easy dataset

ds_root = "../benchmarks/altogether"
ds_name = "easy"
ds_path = os.path.join(ds_root, ds_name)
graph_path = os.path.join(ds_root, ds_name + "_primal_graphs")
print("Visualizing CNF as graphs from:", ds_path)
print("Graphs saving to:", graph_path)

all_items = os.listdir(ds_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

for fn in files:
	print(f"Processing {fn}:")

	# construction
	with open(os.path.join(ds_path, fn)) as f:
		cnf, weights, _ = readCNF(f, mode="MIN")
	cnf_set = [{abs(lit) for lit in clause} for clause in cnf]
	G = construct_primal_graph(cnf_set, len(weights))

	# Stats
	print(f"Nodes: {len(G.nodes())}; Edges: {len(G.edges())}")
	print("Nodes all recorded? ", len(G.nodes()) == len(weights))
	print("Tree width: ", nx.algorithms.approximation.treewidth_min_degree(G)[0])
	
	# visualising the graph
	plt.figure(figsize=(8, 6))
	nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
	plt.title("Graph of CNF")
	plt.savefig(os.path.join(graph_path, fn + ".png"))
	plt.close()
