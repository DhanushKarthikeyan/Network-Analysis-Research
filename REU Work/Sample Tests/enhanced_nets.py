
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import time 

x = time.time()

# Query Parameters
cols = 'distinct(topics_id) as topics'
tbl = 'f34'
modifier = f'order by topics_id'

# --forums = [34,41,77,84] # identified topics
forums = [34]
fig = plt.figure() 
X = nx.Graph()

for forum_id in forums: # iterate through forums
    tdf = get(tbl, cols, None, modifier)
    t = tdf['topics'].tolist() # list of topics

    gdf = get(tbl, 'users_id, topics_id, posts_id', None, None) # entire forum
    #t = t[1:15]
    
    for tp in range(len(t)): #iterate through topics
        print(f'{tp+1} of {len(t)}')
        
        topic = t[tp]
        #pdf = get(tbl, 'users_id as users', f'topics_id = {int(topic)}', 'order by posts_id')
        if tp+1 == 4392:
            print('trying')
            print(gdf[gdf['topics_id'] == topic]['users_id'])

        users = gdf[gdf['topics_id'] == topic]['users_id'].to_numpy()
        # data[data['Value'] == True]

        #users = pdf['users'].to_numpy()
        X.add_nodes_from(users) # add nodes, duplicated nodes are ok

        for i in np.arange(0, users.size):
            for j in np.arange(i+1, users.size):
                if users[i] != users[j]:
                    #X.add_edge(users[i], users[j], weight = 1) # consider removing weighted edge
                    X.add_edge(users[i], users[j])

#nx.draw(X, with_labels = True)

x2 = time.time()
print(f'time elapsed : {x2 - x}')

#plt.show()
#fig.savefig("omg3650", dpi = 500)