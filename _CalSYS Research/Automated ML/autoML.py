
from numpy.lib.function_base import rot90
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lazypredict.Supervised import LazyClassifier, LazyRegressor

z = '3X'
dts = [f'Forum77data_{z}_train.csv']
bts = [f'Forum77data_{z}_test.csv']

data = pd.read_csv(dts[0])
test = pd.read_csv(bts[0])

y_train = data.pop('Class') # Class
X_train = data.drop('Topic',axis = 1)

y_test = test.pop('Class') # Class
X_test = test.drop('Topic',axis = 1)

# fit all models
clf = LazyClassifier(verbose = 0, predictions=True)
# models is a dataframe, predictions are predictions
models, predictions = clf.fit(X_train, X_test, y_train, y_test)    
import seaborn
seaborn.set()
plt.figure()

langs = models.index.tolist()
students = models.loc[:,"F1 Score"].tolist()
plt.bar(x = langs, height = students)
plt.xticks(rotation = 45)
plt.title('Performance of Different Classifiers', fontsize = 16, fontweight = 'bold')
plt.show()
