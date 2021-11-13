

import pandas as pd
import networkx as nx
from networks import make_net, save_net, show_net, get_net
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections
import numpy as np

# Graph
pkp = f'pickleX{12}.p'        
G = get_net(pkp)
G = G.to_undirected()
partition = community_louvain.best_partition(G)
counter = collections.Counter(partition.values())
counter = collections.OrderedDict(sorted(counter.items()))

def get_infected_comm(usrs):
    comms = []
    for u in usrs:
        if u in partition:
            comms.append(partition[u])
    tot_comms = list(set(comms))
    return len(tot_comms), comms

def get_overlap_comm(usrs):
    f,s = [],[]
    mid = len(usrs)//2
    for i in np.arange(0, len(usrs)):
        if i > mid:
            f.append(partition[usrs[i]])
        else:
            s.append(partition[usrs[i]])
    return len(set(f) & set(s))

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def cardinality_matching(users):
    # make a subgraph
    S = G.subgraph(users)
    return nx.algorithms.matching.max_weight_matching(S, maxcardinality=False, weight='weight')

def main():
    users = [3166,3167,3174,3168,3169,727,105,112577]
    #users = [3166,3167,3174,3168,3169,727]
    w = get_overlap_comm(users)
    inf, communities  = get_infected_comm(users)
    #print(f'total communities is {inf}')
    gn = gini_coefficient(np.array(communities))
    gn2 = gini(np.array(communities))

    print(gn)
    print(gn2)

    print(cardinality_matching(users))

if __name__ == "__main__":
    main()


# Number of Infected Communities: # of communities with at least one early adoper
'''
- get list of early adopters
- get list of communities --> sort --> length
'''
# Usage and Adopter Entropy: distribution of adopters across communities
'''
- percentage?
'''
# Fraction of Intra-community user interaction: adopting from the same community / adopting from other community // lower found in early adopters of viral memes
'''
shared communities/other communities
- maybe from root user?
- average fraction?
'''





'''
Baseline measures:
Number of nodes: cardinality of adopters, recent and past users

Avg time to adoption

Structural Density:
Number of Communities

Gini Impurity: behaves similar to entropy for this set

Overlap: between two succesive groups ... doubling?
'''