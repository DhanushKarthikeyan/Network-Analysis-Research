
import pandas as pd
import networkx as nx
from networks import make_net, save_net, show_net, get_net
from networkx.algorithms import community

pkp = f'pickleX{34}.p'        
G = get_net(pkp)
#G.edges.data('weight')
#G.nodes.data()
G = G.to_undirected()

print(len(G.edges.data('weight')))

import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
partition = community_louvain.best_partition(G)
print(partition.values())

# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()