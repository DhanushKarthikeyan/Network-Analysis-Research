

# connection

# need forum name

# get number of topics in forum --> wantRows = True or distinct topics

# iterate through list of topic ID's

    # query to get all users (non-distinct) --> convert column to list

    # iterate through list of chat logs, starting from second index to len(list)

    # add current user and previous user as a connection

    # user ID is preserved across topics


# display Social Network
# get node object metrics

from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import time 

x = time.time()
# Query Parameters
cols = 'distinct(topics_id) as topics'
tbl = 'f12'
modifier = f'order by topics_id'

#forums = [34,41,77,84] # identified topics
forums = [12]
fig = plt.figure()
X = nx.Graph()

for forum_id in forums: # iterate through forums
    where = None
    tdf = get(tbl, cols, where, modifier)
    t = tdf['topics'].tolist()
    #t = t[1:15]
    # get the table into a dataframe

    for topic in t: #iterate through topics
        pdf = get(tbl, 'users_id as users', f'topics_id = {int(topic)}', 'order by posts_id')
        users = pdf['users'].to_numpy()
        X.add_nodes_from(users) # add nodes, duplicated nodes are ok

        # LOGIC to add all edges
        # repeated weight can be removed with DiGraph()
        '''
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if users[i] != users[j]:
                    X.add_edge(users[i], users[j], weight = 1)
        '''

        for i in np.arange(0, users.size):
            for j in np.arange(i+1, users.size):
                if users[i] != users[j]:
                    X.add_edge(users[i], users[j], weight = 1)
        
nx.draw_shell(X, with_labels = True)
x2 = time.time()

print(f'time elapsed : {x2 - x}')

#plt.show()
fig.savefig("Overview", dpi = 500)