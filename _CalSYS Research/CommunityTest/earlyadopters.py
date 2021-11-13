
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import random as rd

def get_early_adopters(forum, alpha, beta):
    casc = {}
    noncasc = {}
    timecsc = {}
    timencsc = {}

    tbl = f'f{forum}'
    gdf = get(tbl, 'users_id, topics_id, posts_id, posted_date', None, None) # entire forum
    # get topics and distinct user count
    tdf = get('t_posts', 'topics_id as topics, count (distinct users_id) as cnt', None, f'where forums_id = {forum} group by topics_id')

    # get all topics that reach n
    qdf = tdf[(tdf['cnt'] >= beta)]['topics'].tolist()
    # get all topics that reach n/2 but not n
    qdf2 = tdf[(tdf['cnt'] >= alpha) & (tdf['cnt'] < beta)]['topics'].tolist() # need to filter further
    
    del tdf
    # get positive user lists for each category
    # get times as well
    
    for t in qdf:
        usrs = gdf[gdf['topics_id'] == t]['users_id'].tolist() # all pos cases
        tms = gdf[gdf['topics_id'] == t]['posted_date'].tolist() # all pos times, need to verify
        index = 0
        casc_users = {}
        lst = []
        lst2 = []
        while len(casc_users) < alpha:
            if usrs[index] not in casc_users.keys(): # if usr != i in list([0])
                casc_users[usrs[index]] = 1 #tms[index] # first alpha users, with their first time
                lst.append(usrs[index])
                lst2.append(tms[index])
            index += 1

        casc[t] = lst
        timecsc[t] = lst2
        #print(f'total in casc is {len(set(usrs))}')
        

    for t2 in qdf2:
        usrs2 = gdf[gdf['topics_id'] == t2]['users_id'].tolist()      #[:alpha]
        tms2 = gdf[gdf['topics_id'] == t2]['posted_date'].tolist()  #[:alpha]
        noncasc_users = {}
        lst3 = []
        lst4 = []
        for z in range(len(usrs2)):
            if len(noncasc_users) < alpha:
                if usrs2[z] not in noncasc_users.keys():
                    noncasc_users[usrs2[z]] = tms2[z] # first alpha users, with their first time
                    lst3.append(usrs2[z])
                    lst4.append(tms2[z])
        
        noncasc[t2] = lst3
        timencsc[t2] = lst4
        #print(f'total in ncasc is {len(set(usrs2))}')
        #avg.append(len(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist())))
        #nonsize[t2] = len(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist()))
        #ns[t2] = list(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist()))
    
    del gdf

    return casc, noncasc, timecsc, timencsc