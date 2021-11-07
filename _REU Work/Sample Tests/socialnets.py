
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
import time 

'''
select distinct(topics_id) as topics from t_posts where forums_id = 1 order by topics_id;
'''
x = time.time()
# Query Parameters
cols = 'distinct(topics_id) as topics'
tbl = 't_posts'
modifier = f'order by topics_id'

#forums = [34,41,77,84] # identified topics
forums = [12]
X = nx.Graph()

for forum_id in forums: # iterate through forums
    where = f'forums_id = {forum_id}'

    tdf = get(tbl, cols, where, modifier)
    t = tdf['topics'].tolist()

    for topic in t: #iterate through topics
        pdf = get(tbl, 'users_id as users', f'forums_id = {forum_id} and topics_id = {int(topic)}', 'order by posts_id')
        users = pdf['users'].tolist()
        X.add_nodes_from(users) # add nodes, duplicated nodes are ok

        # LOGIC to add all edges
        # repeated weight can be removed with DiGraph()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if users[i] != users[j]:
                    X.add_edge(users[i], users[j], weight = 1)
                    pass
        
nx.draw_shell(X, with_labels = True)
#pos = nx.fruchterman_reingold_layout(X)
#nx.draw_networkx(X)

#d = nx.coloring.equitable_color(X, num_colors=20)
#plt.show()
#print(X.degree(weight = 'weight'))
x2 = time.time()

print(f'time elapsed : {x2 - x}')
# =========================================
# Notes:
# - keep entire data within pandas dataframe - more than 191k rows, but 5k queries in SQL can be reduced to 1
# - use numpy for loops, look into xrange instead of range
# reduce size of dataframe returned
# get topics, then filter by topics in main df
#
# ***
# ask about repeated pairs
# ask about sequential posts...back to back
# use pickle to save
# ask about direction of graph...do we want influencer or influenced
# =========================================


# program find communities, users have no shared communities
# louvain community find https://python-louvain.readthedocs.io/en/latest/

# find delta t -- average time elapsed and then filter others out
# gini impurity --> https://python-louvain.readthedocs.io/en/latest/

# find number of frontiers --> page 1

# Look at other paper -------------------
# find centrality of 50 users
# find average centrality
# eigenvector stuff -- in other paper

# up to 20 features across the two papers that I can use

# can try djistrka path as a variable

#neightborhood size ?? maybe size of frontiers


# TRAINING:
# time intermitting ?? --> 
