import time
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import random


hashtag = "4thofjuly"
window = "2018-08-12"

# === TIME EACH STEP ===

# 1. Load CSV
t0 = time.time()
edges = pd.read_csv(f'03_networks/data/windows_{hashtag}/{window}/s2_proj/count_B_imported/edges.csv')
t1 = time.time()
print(f"Load CSV: {t1-t0:.2f}s")

# 2. Build NetworkX graph
t0 = time.time()
G = nx.from_pandas_edgelist(edges, 'source', 'target')
t1 = time.time()
print(f"Build graph: {t1-t0:.2f}s")
print(f"  Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

# Generate walks
walks = []
nodes = list(G.nodes())
for _ in range(3):
    random.shuffle(nodes)
    for node in nodes:
        walk = [node]
        for _ in range(5):
            neighbors = list(G.neighbors(walk[-1]))
            if not neighbors: break
            walk.append(random.choice(neighbors))
        walks.append([str(n) for n in walk])
print(f"Node2Vec init: {t1-t0:.2f}s")

# 4. Train embeddings
t0 = time.time()
# Train
model = Word2Vec(walks, vector_size=32, window=5, min_count=1, workers=1, epochs=1)
t1 = time.time()
print(f"Train embeddings: {t1-t0:.2f}s")

print(f"\nTOTAL: {sum([t1-t0]):.2f}s")