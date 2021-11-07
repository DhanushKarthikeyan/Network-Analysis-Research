import networkx as nx
import numpy as np

G = nx.karate_club_graph()
#G = G.to_undirected()
#G = Z.subgraph([1,2,3,5,6,7,8,9,10,11,12,13])
import community
lp = community.best_partition(G, weight='weight', random_state=40)
print(lp)
j = np.array([users for key, users in lp.items() if users != 0])
print(len([users for key, users in lp.items() if users != 0]))
print(j)
print(np.count_nonzero(j))

modularity2 = community.modularity(lp, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))