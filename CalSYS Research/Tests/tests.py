
'''z = [[17,2,17],[3,4,5],[1,2,4]]

cache = None
cache2 = None
sum = 0
sum2 = 0
for house in z:
    index = 0
    mid = house[0]
    for x in range(1,len(house)):
        if house[x] < mid and mid is not cache:
            mid = house[x]
            index = x 
    sum += mid
    cache = index

for j in range(len(z)):
    house = z[len(z)-1-j]
    index = 0
    mid = house[0]
    for x in range(1,len(house)):
        if house[x] < mid and mid is not cache2:
            mid = house[x]
            index = x 
    sum2 += mid
    cache2 = index

print(min(sum,sum2))'''

def rmdup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

l = [1,2,4,45,56,6,7,2,4,3]
print(rmdup(l))

myDict = {'seven': 7, 'four': 4, 'one': 1, 'two': 2, 'five': 5, 'eight': 8} 

#print(sorted(myDict.iterkeys(), key=lambda k: myDict[k], reverse=True))


temp2 = sorted(myDict, key=myDict.get, reverse=True)
temp = dict(sorted(myDict.items(), key=lambda item: item[1], reverse=True))
print(temp2)


lst = [(1,2),(3,4),(4,5),(6,6)]
for i in lst:
    print(i)
