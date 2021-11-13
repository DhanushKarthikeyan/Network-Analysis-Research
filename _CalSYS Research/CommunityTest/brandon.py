
'''from connect import get
import pandas

# select distinct(users_id) from f77 where topics_id = 164622;


gdf = get('f77', 'distinct(users_id) as users', 'topics_id = 164835', None) # entire forum

users_list = gdf['users'].to_list()

print('[', end = '')
for u in users_list:
    print(f'{u},', end = '')
print(']', end = '')



# [1,2,3,4,5]

# [M,M,W,T,F]

# neighbors , '''


a = [1,2,3,4,5,6,7,8]
b = [7,8,9,10]

com = list(set(b) - set(a))
print(com)



