import numpy as np
import networkx as nx
import cvxpy as cp

edgelist = [(1, 2, 1.0), (2,1,1.0),(2, 3,1.0), (3,2,1.0),(3, 4,3.0), (4,3,3.0),(1,4,2.0),(4,1, 2.0)]
DG = nx.DiGraph()
DG.add_weighted_edges_from(edgelist)
# print(DG)

print(nx.shortest_path(DG, source=1, target=3))

print(DG.edges)

for (u, v, wt) in DG.edges.data('weight'):
    wt = u*v

for (u, v, wt) in DG.edges.data('weight'):
    print(wt)