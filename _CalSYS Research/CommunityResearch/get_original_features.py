
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import datetime as dt
import community

X = None
C = None
UX = None
PR = None

def get_f1(users): # average number of neighbors
    sum = 0
    for usr in users:
        sum += len(list(X.neighbors(usr)))
    return sum/len(users)

def get_f2(root): # NAN for root
    return len(list(X.neighbors(root)))

def get_f3(users): # G.out_degree(1) average
    sum = 0
    for usr in users:
        sum += (X.out_degree(usr))
    return sum/len(users)

def get_f4(pst_tm): # time elapsed
    elapsed = pst_tm[-1] - pst_tm[0]
    return round(elapsed.total_seconds()/60,2)

def get_f5(root): # root user degree centrality --> edges/neighbors
    return nx.degree_centrality(X)[root]

def get_f6(root): # root user out_degree centrality / 
    return nx.out_degree_centrality(X)[root]

def get_f7(root): # root user eigenvector centrality
    return C[root]

def get_f8(root): # cumulative weight of out degree edges
    return X.out_degree(weight = 'weight')[root]

def get_f9(users): # average cumulative weight of out_degree edges
    sum = 0
    for usr in users:
        sum += X.out_degree(weight = 'weight')[usr]
    return sum/len(users)

def get_f10(root): # root user pagerank --> importance ranking
    return PR[root]

def get_f11(users): # average page rank
    sum = 0
    for usr in users:
        sum += PR[usr]
    return sum/len(users)

def get_f12(users): # group out degree centrality
    return nx.group_out_degree_centrality(X, users)

def get_f13(users): # group_betweenness centrality
    return nx.group_betweenness_centrality(X, users, normalized=True, weight='weight')

def get_f14(users): # group closeness centrality - measure of how close the group is to the other nodes in the graph.
    return nx.group_closeness_centrality(X, users, weight='weight')

def get_f15(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*(len(times)-1)),2) # avg and in minutes

def get_communities(users): # num of communities, modularity
    # make subgraph
    S = UX.subgraph(users) # undirected
    lp = community.best_partition(S, weight='weight', random_state=40)
    j = len([users for _, users in lp.items() if users != 0])
    mod = community.modularity(lp, S, weight='weight')
    return j, mod

#get_features(train_threads, train_times, get_net(pkp))
def get_og_features(threads, times, network):
    data = [] # topic_id f1, f2, f3, f4, yes
    
    global X    
    X = network # entire network

    global C 
    C = nx.eigenvector_centrality_numpy(X.reverse(), 'weight') 
    # centrality of all nodes
    # have to reverse graph for out-edges eigenvector centrality

    global UX
    UX = X.to_undirected()

    '''global PR 
    PR = nx.pagerank_numpy(X, alpha=0.9, weight = 'weight')'''
    
    csc = threads['Pos']
    ncsc = threads['Neg']
    tmcsc = times['Pos']
    tmncsc = times['Neg']

    for key, users in csc.items():
        print(f'Doing Forum {key}')
        
        f1 = round(get_f1(users),2) # neighbors
        f2 = round(get_f2(users[0]),2) # NAN root
        f3 = round(get_f3(users),2) # out degree, shows propogation of influence
        f4 = get_f4(tmcsc[key]) # users, users_time -> time elapsed
        f5 = round(get_f5(users[0]),2) # rootuser, degree centrality
        f6 = round(get_f6(users[0]),2) # rootuser, out_degree_centrality
        f7 = round(get_f7(users[0]),2) # rootuser eigenvector centrality
        f8 = round(get_f8(users[0]),2) # cumulative weight of out degree edges
        f9 = round(get_f9(users),2) # average cumulative weight of out_degree edges
        #f10 = round(get_f10(users[0]),2) # root user pagerank
        #f11 = round(get_f11(users),2) # avg page rank
        f12 = round(get_f12(users),2) # group out degree centrality
        #f13 = round(get_f13(users),2) # group_betweenness centrality
        f14 = round(get_f14(users),2) # group closeness centrality
        f15 = get_f15(tmcsc[key]) # avg time to adoption

        f16, f17 = get_communities(users) # num of communities, modularity
        data.append([f'Topic{key}',f1,f2,f3,f4,f5,f6,f7,f8,f9,f12,f14,f15,f16,f17,1]) # topic_id f1, f2, f3, f4... yes
        #make_subgraph(value)

    for key, users in ncsc.items():
        print(f'Doing Forum {key}')
        
        f1 = round(get_f1(users),2) # neighbors
        f2 = round(get_f2(users[0]),2) # NAN root
        f3 = round(get_f3(users),2) # out degree, shows propogation of influence
        f4 = get_f4(tmncsc[key]) # users, users_time -> time elapsed
        f5 = round(get_f5(users[0]),2) # rootuser, degree centrality
        f6 = round(get_f6(users[0]),2) # rootuser, out_degree_centrality
        f7 = round(get_f7(users[0]),2) # rootuser eigenvector centrality
        f8 = round(get_f8(users[0]),2) # cumulative weight of out degree edges
        f9 = round(get_f9(users),2) # average cumulative weight of out_degree edges
        #f10 = round(get_f10(users[0]),2) # root user pagerank
        #f11 = round(get_f11(users),2) # avg page rank
        f12 = round(get_f12(users),2) # group out degree centrality
        #f13 = round(get_f13(users),2) # group_betweenness centrality
        f14 = round(get_f14(users),2) # group closeness centrality
        f15 = get_f15(tmncsc[key]) # avg time to adoption

        f16, f17 = get_communities(users) # avg time to adoption
        data.append([f'Topic{key}',f1,f2,f3,f4,f5,f6,f7,f8,f9,f12,f14,f15,f16,f17,0]) # topic_id f1, f2, f3, f4... no

    pdf = pd.DataFrame(data, columns=['Topic', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6','F7', 'F8', 'F9', 'F12', 'F14', 'F15', 'F16', 'F17', 'Class'])
    pdf.fillna(0, inplace=True) # clean dataframe
    return pdf
