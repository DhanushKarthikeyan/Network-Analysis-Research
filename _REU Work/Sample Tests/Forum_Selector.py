
'''
select topics_id, count(distinct(users_id)) from t_posts where forums_id = 1 group by topics_id having count(distinct(users_id)) > 5 order by topics_id
'''


import psycopg2
from connect import get

import pandas as pd
from matplotlib import pyplot as plt

# Query Parameters
df = None
cols = 'topics_id, count(distinct(users_id)) as x'
tbl = 't_posts'
cnt = 40
modifier = f'group by topics_id having count(distinct(users_id)) > {cnt} order by topics_id'
modifier2 = f'group by topics_id having count(distinct(users_id)) > {cnt*2} order by topics_id'
modifier3 = f'group by topics_id having count(distinct(users_id)) > {cnt*3} order by topics_id'

x, w, y, z = [], [], [], []
ideal = {}

#def get(table_name, cols='*', where=None, modifier=None):
for id in range(1, 94): # 93 forums
    
    where = f'forums_id = {id}'

    df, count = get(tbl, cols, where, modifier, True)
    df, count2 = get(tbl, cols, where, modifier2, True)
    df, count3 = get(tbl, cols, where, modifier3, True)

    x.append(id)
    w.append(count)
    y.append(count2)
    z.append(count3)
    

chart = True
if chart:
    fig = plt.figure()
    plt.plot(x, w, label = f'above {cnt}', color = 'red')
    plt.plot(x, y, label = f'above {cnt*2}', color = 'blue')
    plt.plot(x, z, label = f'above {cnt*3}', color = 'green')
    plt.xlabel('Forums ID')
    plt.ylabel('Number of Remaining Topics')
    plt.title('Data Distribution across all Topics in Forums')
    plt.legend()

    save = False
    if not save:
        plt.show()
    else:
        fig.savefig("IdealForums", dpi = 500)
        print('Image Saved to Local directory!')


isolate = True
if isolate:
    # To isolate ideal forums
    dt1 = dict(zip(x, z))
    dt2 = {}
    for key, value in dt1.items():
        if value > 3:
            dt2[key] = value

    #print(str(dt2))
    for key, value in dt2.items():
        print(f'Forum {key} has {value} topics at least with {cnt*3} users')


'''
overlay: https://stackoverflow.com/questions/751567/both-a-top-and-a-bottom-axis-in-pylab-e-g-w-different-units-or-left-and-rig
'''
