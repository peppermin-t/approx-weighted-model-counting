import os
import networkx as nx
import matplotlib.pyplot as plt

from utils import readCNF, construct_dual_graph, dfs_all_components

    
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
	G = construct_dual_graph(cnf_set)

	# Stats
	print(f"Nodes: {len(G.nodes())}; Edges: {len(G.edges())}")
	print("Nodes all recorded? ", len(G.nodes()) == len(cnf))
	print("Tree width: ", nx.algorithms.approximation.treewidth_min_degree(G)[0])
	
	# visualising the graph
	pos = nx.spring_layout(G)
	plt.figure(figsize=(16, 12))
	nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=12, edge_color='gray')
	plt.title("Graph of CNF", fontsize=16)
	plt.savefig(os.path.join(graph_path, fn + ".png"))
	plt.close()
	
	# visualising dfs marked graph with same layout
	order = dfs_all_components(G)
	colored_paths = []
	for i in range(1, len(order)):
		if G.has_edge(order[i - 1], order[i]):
			colored_paths.append((order[i - 1], order[i]))
	
	print(f"dfs connected edges: {len(colored_paths)}")

	edge_colors = []
	for edge in G.edges():
		if edge in colored_paths or (edge[1], edge[0]) in colored_paths:
			edge_colors.append('red')
		else:
			edge_colors.append('gray')
   
	plt.figure(figsize=(16, 12))
	nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, font_size=12)
	plt.title("DFS Marked Dual Graph of CNF", fontsize=16)
	plt.savefig(os.path.join(graph_path, fn + "_dfs.png"))
	plt.close()
 
	# visualising natural order marked graph with same layout
	colored_paths = []
	for i in range(1, len(cnf)):
		if G.has_edge(i - 1, i) or G.has_edge(i, i - 1):
			colored_paths.append((i - 1, i))

	print(f"natural order connected edges: {len(colored_paths)}")
	
	edge_colors = []
	for edge in G.edges():
		if edge in colored_paths or (edge[1], edge[0]) in colored_paths:
			edge_colors.append('red')
		else:
			edge_colors.append('gray')
   
	plt.figure(figsize=(16, 12))
	nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, font_size=12)
	plt.title("DNatural Order Marked Dual Graph of CNF", fontsize=16)
	plt.savefig(os.path.join(graph_path, fn + "_nodfs.png"))
	plt.close()
