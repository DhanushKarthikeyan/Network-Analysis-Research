'''
import networkx as nx 
import matplotlib.pyplot as plt
from networkx.algorithms import approximation
  
G = nx.Graph()
  
plt.figure(figsize =(9, 12))
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), 
                         (4, 5), (4, 6), (5, 7), (5, 8), (7, 8)])
  
# original Graph created
plt.subplot(211)
print("The original Graph:")
  
nx.draw_networkx(G)


H = G.subgraph([1, 2, 3, 4])
# [1, 2, 3, 4] is the subset of 
# the original set of nodes
  
plt.subplot(212)
print("The Subgraph:")
nx.draw_networkx(H)
#plt.show()

c = approximation.clustering_coefficient.average_clustering(G, 1000000, 1)
print(c)
'''

'''

dt = {}

x = [1,2,3,4,5]
y = ['a', 'b', 'c', 'd', 'e']

for i in x:
    dt[str(y[x-1])].append(list(i))

print(dt)


from itertools import repeat

#new_list.extend(repeat(given_value,5))

lst = []
lst.extend(repeat('hello',5))
print(lst)
'''  
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


G = nx.DiGraph() # DiGraph() if duplicate pairs should not be counted
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)

G.add_edge(1, 2, weight = 1)
G.add_edge(1, 3, weight = 1)
G.add_edge(1, 3, weight = 2) # overwrite edge
G.add_edge(1, 4, weight = 1)
G.add_edge(4, 1, weight = 1)
G.add_edge(5, 2, weight = 1)
G.add_edge(6, 3, weight = 1)
G.add_edge(7, 3, weight = 2) # overwrite edge
G.add_edge(7, 4, weight = 1)
G.add_edge(7, 1, weight = 1)

#G.add_node(1)
#G.add_edge(1, 1, weight = 1)

#G.add_nodes_from([6,7,8,9])

#print(G.nodes.data())

print(G.out_degree(weight = 'weight')[1]) #influence - total edges
print(G.degree(weight = 'weight')[1]) #influence - total edges

#print(len(list(G.neighbors(1))))
#print(len(list(G.neighbors(6))))
#print(list(G.successors(1)))
print(G.degree(1))
#print(G.in_degree(1))
#print(G.out_degree(1))
#print(G[1][3]['weight'])
#nx.draw_shell(G, with_labels = True)
#plt.show()
dc = nx.degree_centrality(G)
print(dc[1])

# make an undirected copy of the digraph
G = G.to_undirected()

import community
lp = community.best_partition(G, weight='weight')
print(lp)
j = np.array([users for key, users in lp.items()])
print(j)
print(np.count_nonzero(j))
modularity2 = community.modularity(lp, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))


'''centrality = nx.eigenvector_centrality(G)
x = (sorted((v, f"{c:0.2f}") for v, c in centrality.items()))
print(x[0])'''

'''centrality = nx.eigenvector_centrality_numpy(G.reverse(), 'weight')
for node in centrality: 
    print(node)
    print(centrality[node])'''
#print([f"{node} {centrality[node]:0.2f}" for node in centrality])
'''sum = 0
for i in range(1,5):
    sum += nx.out_degree_centrality(G)[i]
    G.out_degree(i)
sum = sum/4
print(f'sum is {sum}')

print(nx.group_in_degree_centrality(G, [1,2,3])) #has to be less than total nodes
print(nx.group_degree_centrality(G, [2,3,4])) #has to be less than total nodes
print(nx.group_betweenness_centrality(G, [7], normalized=True, weight = 'weight')) #has to be less than total nodes
print(nx.group_closeness_centrality(G, [1], weight='weight'))
'''


'''PR = nx.pagerank_numpy(G, alpha=0.9, weight = 'weight')
print(PR[1])'''

'''
x = [1,2,3,4]

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        print(x[i], x[j])

print('done')

'''


'''
import numpy as np
users = np.array([1,2,3,4])

for i in np.arange(0, users.size):
    for j in np.arange(i+1, users.size):
        if users[i] != users[j]:
            print(f' {users[i]} and {users[j]}')
            '''
'''
users2 = [1,2,3,4]
for i in range(len(users2)):
    for j in range(i + 1, len(users2)):
        if users2[i] != users2[j]:
            #print(f' {users[i]} and {users[j]}')
            pass



# Theoretical 
import networkx as nx
nodes = [1,2,3,4,5]

G = nx.DiGraph()
G.add_nodes_from(nodes)

for i in range(len(nodes)-1):
    G.add_weighted_edges_from([(nodes[i],node, 1) for node in nodes[i+1:]])

print(G.nodes.data())
print(G.degree(weight = 'weight'))

print(G.out_degree(2))
'''
