
import psycopg2
from connect import get

import pandas as pd
from matplotlib import pyplot as plt

'''
/* Query to break down total posts and unique users by forum */
    Select forums_id, 
        count(posts_id) as TotalPosts, 
        count(distinct users_id) as UniqueUsers 
    from t_posts 
        group by forums_id order by forums_id;
'''

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
plt.xlabel('Forums ID')
plt.ylabel('Total Activity')
plt.title('Data Distribution across all Forums')
plt.legend()

save = False
if not save:
    plt.show()
else:
    fig.savefig("Overview", dpi = 500)
    print('Image Saved to Local directory!')


'''
overlay: https://stackoverflow.com/questions/751567/both-a-top-and-a-bottom-axis-in-pylab-e-g-w-different-units-or-left-and-rig
'''