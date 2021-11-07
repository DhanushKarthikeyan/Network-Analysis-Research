
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

z = '3X'
dts = [f'../Forum34data_{z}_train.csv',f'../Forum41data_{z}_train.csv',f'../Forum77data_{z}_train.csv',f'../Forum84data_{z}_train.csv']
bts = [f'../Forum34data_{z}_test.csv',f'../Forum41data_{z}_test.csv',f'../Forum77data_{z}_test.csv',f'../Forum84data_{z}_test.csv']
avg = 0
for z in range(len(dts)):
    #data = pd.read_csv(frm)
    print(dts[z])
    data = pd.read_csv(dts[z])
    test = pd.read_csv(bts[z])

    train_label = data.pop('Class') # Class
    train_data = data.drop('Topic',axis = 1)

    test_label = test.pop('Class') # Class
    test_data = test.drop('Topic',axis = 1)

    seed = 60 
    # splitting
    #train_data, test_data, train_label, test_label = train_test_split(x,y,test_size=0.20, random_state = seed)

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
    
    model.fit(train_data, train_label)

    #pipe = make_pipeline(col_trans, rf_classifier)
    #pipe.fit(train_data, train_label)

    y_pred = model.predict(test_data)

    #print(model.feature_importances_) # UseThese

    #X_train_encoded = encode_and_bind(train_data, features_to_encode)

    #--------------------------------------------------------

    '''accuracy = accuracy_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    precision = precision_score(test_label, y_pred)'''

    #print(f"The accuracy of the model is {round(accuracy,3)*100} %")
    #print(f"The recall of the model is {round(recall,3)*100} %")
    #print(f"The precision of the model is {round(precision,3)*100} %")
    print(f"The f1 is {round(f1_score(test_label, y_pred),3)}")
    avg += round(f1_score(test_label, y_pred),3)
    #print(f'it is {(2*precision*recall)/(precision+recall)}')
print(f'average is {avg/len(bts)}')
