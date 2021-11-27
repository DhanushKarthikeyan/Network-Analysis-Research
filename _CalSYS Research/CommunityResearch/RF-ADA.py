
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import itertools
#model = AdaBoostClassifier(random_state=50, base_estimator= rf_classifier, n_estimators=100, learning_rate= 0.001)
from sklearn.ensemble import AdaBoostClassifier

'''z = '10X'
dts = [f'Forum77data_{z}_train.csv']
bts = [f'Forum77data_{z}_test.csv']

data = pd.read_csv(dts[0])
test = pd.read_csv(bts[0])

data.fillna(0)
test.fillna(0)

y_train = data.pop('Class') # Class
X_train = data.drop('Topic',axis = 1)

y_test = test.pop('Class') # Class
X_test = test.drop('Topic',axis = 1)'''

rf_classifier = RandomForestClassifier(
    # min_samples_leaf=30,
    n_estimators=900,
    #bootstrap=True,
    #oob_score=True,
    #n_jobs=-1,
    random_state=0,
    criterion ='gini'
    )
model = AdaBoostClassifier(random_state=50, base_estimator= rf_classifier, n_estimators=100, learning_rate= 0.001)

model.fit(X_train, y_train)
#pipe = make_pipeline(col_trans, rf_classifier)
#pipe.fit(train_data, train_label)
y_pred = model.predict(X_test)

    #print(model.feature_importances_) # UseThese

    #X_train_encoded = encode_and_bind(train_data, features_to_encode)

    #--------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"The accuracy of the model is {round(accuracy,3)*100} %")
print(f"The recall of the model is {round(recall,3)*100} %")
print(f"The precision of the model is {round(precision,3)*100} %")
print(f"The f1 is {round(f1_score(y_test, y_pred),3)}")