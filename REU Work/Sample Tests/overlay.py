
import psycopg2
from connect import get

import pandas as pd
from matplotlib import pyplot as plt

df = None
cols = 'forums_id, count(posts_id) as TotalPosts, count(distinct users_id) as UniqueUsers'
tbl = 't_posts'
where = None
modifier = 'group by forums_id order by forums_id'

#def get(table_name, cols='*', where=None, modifier=None):
df = get(tbl, cols, where, modifier)
x = df['forums_id'].tolist()
y = df['UniqueUsers'].tolist()
z = df['TotalPosts'].tolist()
z[:] = [x / 40 for x in z] #scaling factor list comprehension

fig = plt.figure()
plt.plot(x, y, label = 'Unique Users')
plt.plot(x, z, label = 'Total Posts*40')
plt.title('Number of Unique Users and Total Posts per Forum')
plt.legend()

save = False
if not save:
    plt.show()
else:
    fig.savefig("Overview", dpi = 500)
    print('Image Saved to Local directory!')


'''
- get thread level granularity
- find few forums with threads that qualify
- get initial 50
- use python to build network
- finds communities over time
- look at Ruicheng's features, but cant use since they're cascade-based
- cascade-based is our baseline
    - user based is after (hacker adoption)
    - time constraints (spans --> dynamic)
    - retrain model on the fly
- we do cascade approach first
- user_ids are constrained within the forum --> cannot extrapolate network to other forums
- extract features from network to determine cascade



'''


'''
overlay: https://stackoverflow.com/questions/751567/both-a-top-and-a-bottom-axis-in-pylab-e-g-w-different-units-or-left-and-rig
'''