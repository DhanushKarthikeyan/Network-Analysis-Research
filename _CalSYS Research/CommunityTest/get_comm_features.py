

import pandas as pd
import networkx as nx
from networks import make_net, save_net, show_net, get_net
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections
import numpy as np
import datetime as dt

X = None
G = None
partition = None

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

def cardinality_matching(users):
    # make a subgraph
    S = G.subgraph(users)
    return nx.algorithms.matching.max_weight_matching(S, maxcardinality=False, weight='weight')

def get_avgtime(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*(len(times)-1)),2) # avg and in minutes

#get_features(train_threads, train_times, get_net(pkp))
def get_features(threads, times, network):
    data = [] # topic_id f1, f2, f3, f4, yes
    
    global X    
    X = network # entire network

    global G
    G = X.to_undirected()

    global partition
    partition = community_louvain.best_partition(G)
    
    csc = threads['Pos']
    ncsc = threads['Neg']
    tmcsc = times['Pos']
    tmncsc = times['Neg']

    print('Doing Yes Cases')
    for key, users in csc.items():
        print(f'Doing Forum {key}')

        f1 = get_overlap_comm(users)
        f2, comm  = get_infected_comm(users)
        f3 = round(gini_coefficient(np.array(comm)),2)
        #f4 = cardinality_matching(users)
        f4 = get_avgtime(tmcsc[key]) # users, users_time -> time elapsed

        data.append([f'Topic{key}',f1,f2,f3,f4,1]) # topic_id f1, f2, f3, f4... yes

    print('Doing No Cases')
    for key, users in ncsc.items():
        print(f'Doing Forum {key}')
        
        f1 = get_overlap_comm(users)
        f2, comm  = get_infected_comm(users)
        f3 = round(gini_coefficient(np.array(comm)),2)
        #f4 = cardinality_matching(users)
        f4 = get_avgtime(tmncsc[key]) # users, users_time -> time elapsed

        data.append([f'Topic{key}',f1,f2,f3,f4,0]) # topic_id f1, f2, f3, f4... yes

    pdf = pd.DataFrame(data, columns=['Topic', 'F1', 'F2', 'F3', 'F4', 'Class'])
    return pdf
