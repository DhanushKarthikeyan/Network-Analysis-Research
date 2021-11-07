
from connect import get
import pandas as pd
from matplotlib import pyplot as plt

'''
** Sample Query which we iterate **
select count(*) from test where uniqueusers > 9;

**
select count(forums_id), avg(totalposts), avg(uniqueusers) from test where uniqueusers > 2; 
'''

df = None #empty dataframe
cols = 'count(forums_id) as remaining'
tbl = 'test'
modifier = None

thr, thr2, rem, rem2 = [],[],[],[]

for i in range(0,90):
    # def get(table_name, cols='*', where=None, modifier=None):
    where = f'uniqueusers > {i}'
    df = get(tbl, cols, where, modifier)
    
    # Inverse traversal of query
    where2 = f'uniqueusers > {150-i}'
    df2 = get(tbl, cols, where2, modifier)

    rem.append(df['remaining'].tolist()[0])
    rem2.append(df2['remaining'].tolist()[0])
    thr.append(i)
    thr2.append(150-i)

fig = plt.figure()
plt.plot(thr, rem, label = 'Lower threshold', color = 'blue')
plt.plot(thr2, rem2, label = 'Upper threshold', color = 'red')
plt.xlabel('Threshold Value')
plt.ylabel('Applicable Forums')
plt.title('Threshold Shifting vs Remaining Forums')
plt.legend()

save = False
if not save:
    plt.show()
else:
    fig.savefig("Thresholds", dpi = 500)
    print('Image Saved to Local directory!')



'''
overlay: https://stackoverflow.com/questions/751567/both-a-top-and-a-bottom-axis-in-pylab-e-g-w-different-units-or-left-and-rig
'''