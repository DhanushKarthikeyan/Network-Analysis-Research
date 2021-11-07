
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.utils import shuffle
import random
import itertools

data = pd.read_csv('Forum77data.csv')
#data = shuffle(data)

y = data.pop('Class') # Class
x = data.drop('Topic',axis = 1)
x = x.drop('F3',axis = 1)
x = x.drop('F4',axis = 1)
x = x.drop('F2',axis = 1)
x = x.drop('F1',axis = 1)


res = random.sample(range(100), 7)
x['newF'] = res
seed = 50
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state = seed)
print(y_test.size)


#print(X_train.dtypes)
#features_to_encode = X_train.columns[X_train.dtypes==object].tolist()  # convert to sparse matrix

'''
col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )
'''
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')


pipe = make_pipeline(rf_classifier)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"The accuracy of the model is {round(accuracy,5)*100} %")
print(f"The recall of the model is {round(recall,3)*100} %")
print(f"The precision of the model is {round(precision,5)*100} %")