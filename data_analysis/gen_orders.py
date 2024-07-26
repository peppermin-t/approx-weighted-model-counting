import os
import pickle
import networkx as nx

ds_root = "../benchmarks/altogether"
ds_name = "easy"
print("Generating orders from dataset class:", ds_name)
graph_path = os.path.join(ds_root, ds_name + "_primal_graphs")

all_items = os.listdir(graph_path)
files = [fn for fn in all_items if os.path.isfile(os.path.join(graph_path, fn)) and fn.endswith('.pkl')]

for fn in files:
	print(f"Processing {fn}:")
	with open(os.path.join(graph_path, fn), "rb") as f:
		G = pickle.load(f)

	dfs_nodes = list(nx.dfs_preorder_nodes(G, source=0))
	print(dfs_nodes)