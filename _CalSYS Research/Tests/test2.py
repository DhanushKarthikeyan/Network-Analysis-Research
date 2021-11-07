'''
import time

x1 = time.time()
d = {'x': 1, 'y': 2, 'z': 3} 
for key in d:
    print (key, 'corresponds to', d[key])
x2 = time.time()

for key, value in d.items():
    print (key, 'corresponds to', value)
x3 = time.time()

print((x2 - x1)*1000)
print((x3 - x2)*1000)

'''
'''
print(round(111.333333, 3))

print("Total students : %3d, Boys : %.2f" % (240, 120))
'''
import numpy as np
import time

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

def gini3(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

v = np.random.rand(1000)
x1 = time.time()
print(gini(v))
x2 = time.time()
print(gini3(v))
x3 = time.time()

print(x2-x1)
print(x3-x2)

# ===================== root user =========================
''' - root degree centrality
    - root out degree centrality
    - root out-edge eigenvector centrality
    - cumulative out degree weight
    - root Katz centrality      *
    - root user pagerank_numpy
'''
# ===================== early adopters ====================
''' - avg number of neighbors
    - avg out degree
    - time elapsed
    - avg out degree weight sum
    - average page rank
    - group out degree centrality
    - group_betweenness centrality
    - group closeness centrality
    - average time to adoption    
'''
# ===================== communities =======================
''' - 
    - 
    - 
'''
# =========================================================


# nx.betweenness_centrality(graph)
# nx.closeness_centrality(graph)


# closeness cetnrality
# betweeeness centrality

# betweeness centrality ** --> shortest path



# wait for community till later
# run lovain 3 times for insitial adopters, innovators, 3rd