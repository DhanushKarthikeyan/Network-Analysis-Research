from matplotlib import pyplot as plt
import pandas as pd

tdf = pd.read_csv('Forum34data_3x_train.csv')

#plt.boxplot( df['Class'].tolist(),df['F15'].tolist())

qdf = tdf[(tdf['Class'] == 1)]['F15'].tolist()
wdf = tdf[(tdf['Class'] == 0)]['F15'].tolist()
n = [i for i in range(len(qdf))]

fig = plt.figure()
plt.plot(n, qdf, label = 'positive class')
plt.plot(n, wdf, label = 'negative class')
#plt.plot(x, z, label = 'Total Posts*40')
plt.title('Avg Adoption Time as a Baseline')
plt.xlabel('Cascade ID')
plt.ylabel('Average Time to Adoption')
plt.legend()
plt.show()