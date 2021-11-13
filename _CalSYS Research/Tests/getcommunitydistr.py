

import pandas as pd
import networkx as nx
from networks import make_net, save_net, show_net, get_net
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections

forums = [34,41,77,84]
cp = ['red','blue','green','black']
fig = plt.figure()

for z in range(len(forums)):
    pkp = f'pickleX{forums[z]}.p'        
    G = get_net(pkp)
    G = G.to_undirected()
    print(len(G.edges.data('weight')))

    partition = community_louvain.best_partition(G)
    
    counter = collections.Counter(partition.values())
    counter = collections.OrderedDict(sorted(counter.items()))
    
    plt.plot(counter.keys(), counter.values(), color = cp[z])
plt.title("Community Distribution")
#plt.ylim(-50,50)
plt.xlabel("Community Label")
plt.ylabel("User Count")
plt.show()



