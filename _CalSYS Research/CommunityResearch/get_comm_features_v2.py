

import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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

def find_frontiers(users, times):
    lmda = 3000 # lamda value
    part = 1
    for t in range(1, len(times)):
        l = int((times[t] - times[0]).total_seconds()/60)

        if l > 300:
            #print(f'part was {t} out of {len(times)}')
            part = t
            break
    f = users[:part]
    fn= users[part:]
    return f, fn

def get_overlap(comm1, comm2):
    return len(set(comm1) & set(comm2))

def gini_coefficient(x): # defer to ruicheng's equation
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def get_avgtime(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*(len(times)-1)),2) # avg and in minutes

def get_features(threads, times, network):
    data = []
    
    global X    
    X = network 

    global G
    G = X.to_undirected()

    # Find Communities
    global partition
    partition = community_louvain.best_partition(G)
    
    csc = threads['Pos']
    ncsc = threads['Neg']
    tmcsc = times['Pos']
    tmncsc = times['Neg']

    #print('Doing Positive Cases')
    for key, users in csc.items():
        #print(f'Doing Forum {key}')

        # find frontiers
        F, Fn = find_frontiers(users, tmcsc[key])

        # find communities K
        KV, KV_comm = get_infected_comm(users)
        KF, KF_comm = get_infected_comm(F)
        KFn, KFn_comm = get_infected_comm(Fn)

        # find gini impurity of communities
        GKV = round(gini_coefficient(np.array(KV_comm)),2)
        GKF = round(gini_coefficient(np.array(KF_comm)),2)
        GKFn = round(gini_coefficient(np.array(KFn_comm)),2)

        # find overlap between communities
        OKV_KF  = get_overlap(KV_comm, KF_comm)
        OKV_KFn = get_overlap(KV_comm, KFn_comm)
        OKF_KFn = get_overlap(KF_comm, KFn_comm)

        # find average time
        avg = int(get_avgtime(tmcsc[key]))

        data.append([f'Topic{key}',KV,KF,KFn,GKV,GKF,GKFn,OKV_KF,OKV_KFn,OKF_KFn,avg,1])

    #print('Doing Negative Cases')
    for key, users in ncsc.items():
        #print(f'Doing Forum {key}')

        # find frontiers
        F, Fn = find_frontiers(users, tmncsc[key])

        # find communities K
        KV, KV_comm = get_infected_comm(users)
        KF, KF_comm = get_infected_comm(F)
        KFn, KFn_comm = get_infected_comm(Fn)

        # find gini impurity of communities
        GKV = round(gini_coefficient(np.array(KV_comm)),2)
        GKF = round(gini_coefficient(np.array(KF_comm)),2)
        GKFn = round(gini_coefficient(np.array(KFn_comm)),2)

        # find overlap between communities
        OKV_KF  = get_overlap(KV_comm, KF_comm)
        OKV_KFn = get_overlap(KV_comm, KFn_comm)
        OKF_KFn = get_overlap(KF_comm, KFn_comm)

        # find average time
        avg = int(get_avgtime(tmncsc[key]))

        data.append([f'Topic{key}',KV,KF,KFn,GKV,GKF,GKFn,OKV_KF,OKV_KFn,OKF_KFn,avg,0])

    pdf = pd.DataFrame(data, columns=['Topic','KV','KF','KFn','GKV','GKF','GKFn','OKV_KF','OKV_KFn','OKF_KFn','elapsed','Class'])
    pdf.fillna(0, inplace=True)
    return pdf

# alpha 2->10
# feature ranking
# compare ruicheng to my features
