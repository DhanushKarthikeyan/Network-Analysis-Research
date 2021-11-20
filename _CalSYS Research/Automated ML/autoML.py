
from numpy.lib.function_base import rot90
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lazypredict.Supervised import LazyClassifier, LazyRegressor
import seaborn

from datetime import datetime
today = datetime.now()

frms = ['77']
mult = ['10X']
for forum in frms:
    for m in mult:
        dts = [f'Forum{forum}data_{m}_train.csv']
        bts = [f'Forum{forum}data_{m}_test.csv']

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
        
        #seaborn.set()
        fig = plt.figure()

        mdls = models.index.tolist()
        f1_score = models.loc[:,"F1 Score"].tolist()

        thr = 0.7
        #mdls, f1_score = zip(*[(a, b) for a, b in zip(mdls, f1_score) if b>thr]) #only above 70% f1 score

        plt.bar(x = mdls, height = f1_score)
        plt.xticks(rotation = 70)
        plt.tick_params(axis='x', which='major', labelsize=7)
        plt.title(f'Performance of Different Classifiers in Forum{forum} {m} above {thr}', fontsize = 16, fontweight = 'bold')
        #plt.show()
        dt_string = today.strftime("%d-%m-hour%H")
        fig.savefig(f"F{forum}_{m}_{dt_string}", dpi = 500, bbox_inches='tight')
