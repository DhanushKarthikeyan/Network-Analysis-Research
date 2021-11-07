
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk

def make_net(forums):

    # Query Parameters
    cols = 'distinct(topics_id) as topics'
    tbl = f'f{forums}'
    modifier = f'order by topics_id'
 
    X = nx.DiGraph()

    for forum_id in [forums]: # iterate through forums
        tdf = get(tbl, cols, None, modifier) # get all distinct topics
        t = tdf['topics'].tolist() # list of topics
        gdf = get(tbl, 'users_id, topics_id, posts_id, posted_date', None, None) # entire forum
        
        for tp in range(len(t)): #iterate through topics
            print(f'{tp+1} of {len(t)}')
            users = gdf[gdf['topics_id'] == t[tp]]['users_id'].to_numpy()

            X.add_nodes_from(users) # add nodes, duplicated nodes are ok

            for i in np.arange(0, users.size):
                for j in np.arange(i+1, users.size):
                    if users[i] != users[j]: #avoid cycles
                        #X.add_edge(users[i], users[j], weight = 1) # consider removing weighted edge
                        if X.has_edge(users[i], users[j]):
                            # we added this one before, just increase the weight by one
                            X[users[i]][users[j]]['weight'] += 1
                        else:
                            # new edge. add with weight=1
                            X.add_edge(users[i], users[j], weight=1)
    return X

def save_net(N, forum_id):
    path = f"pickleX{forum_id}.p"
    with open(path, 'wb') as f:
        pk.dump(N, f)
    print(f'Pickle file saved for Forum {forum_id} at {path}...')
    return path

def show_net(path, forum, save = False):
    fig = plt.figure() 
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')

    nx.draw_shell(load, with_labels = True)
    plt.show()
    if save:
        fig.savefig(f"Forum{forum} Network", dpi = 500)
    return 0

def get_net(path):
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')
    return load


# ================================================================================================



# create driver code 
# pass forum_id into network code to construct entire network

# pass df back and iterate through
# make a code that checks forum with topics size n vs size n/2
    # equal values of each
    # bigger forum user count - data is more significant
    # more forums
# run query to retrieve topics id's where above > size and > size/2
# iterate through each topic and find first n/2 distinct users in that topic
    # feature 1
    # feature 2
    # feature 3
    # NAN, PNE, Time Elapsed, Loeveine
    # if  >= n then yes else no
    # write to dataframe or csv
        # preferably df.to_csv()


# alternate: create hierachy chart of data
    # forum --> table --> post
# create example of cascade vs noncascade
    # go thru previous papers sample graphs
# generate sample network
