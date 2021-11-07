import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

z = '3X'
dts = [f'../Forum34data_{z}_train.csv',f'../Forum77data_{z}_train.csv',f'../Forum84data_{z}_train.csv']
bts = [f'../Forum34data_{z}_test.csv',f'../Forum77data_{z}_test.csv',f'../Forum84data_{z}_test.csv']
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

    seed = 50 
    #train_data, test_data, train_label, test_label = train_test_split(data,label,test_size=0.2, random_state = seed)


    features_to_encode = train_data.columns[train_data.dtypes==object].tolist()  

    col_trans = make_column_transformer(
                            (OneHotEncoder(handle_unknown='ignore'),features_to_encode),
                            remainder = "passthrough"
                            )

    nb_model = GaussianNB()


    pipe = make_pipeline(col_trans, nb_model)
    pipe.fit(train_data, train_label)

    y_pred = pipe.predict(test_data)

    def encode_and_bind(original_dataframe, features_to_encode):
        dummies = pd.get_dummies(original_dataframe[features_to_encode])
        res = pd.concat([dummies, original_dataframe], axis=1)
        res = res.drop(features_to_encode, axis=1)
        return(res)

    accuracy = accuracy_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    precision = precision_score(test_label, y_pred)
    f1 = f1_score(test_label, y_pred)

    #print("Naive Bayes")
    print(f"The accuracy of the model is {round(accuracy,3)*100} %")
    print(f"The recall of the model is {round(recall,3)*100} %")
    print(f"The precision of the model is {round(precision,3)*100} %")
    print(f"The f1 of the model is {round(f1,3)}")

    f1 = f1_score(test_label, y_pred)
    #print(f"The f1 of the model is {round(f1,3)}")
    avg += round(f1,3)
    #print(f'it is {(2*precision*recall)/(precision+recall)}')
print(f'average is {avg/len(bts)}')

