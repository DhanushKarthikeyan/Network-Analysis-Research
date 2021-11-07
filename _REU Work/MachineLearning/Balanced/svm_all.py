import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import itertools
from sklearn import svm

'''frms = ['../Forum34data.csv','../Forum41data.csv','../Forum77data.csv','../Forum84data.csv']
avg = 0
for frm in frms:'''
data = pd.read_csv('../Forum34data.csv')
print(data)
label = data.pop('Class') # Class
data = data.drop('Topic',axis = 1)
seed = 30#50 
train_data, test_data, train_label, test_label = train_test_split(data,label,test_size=0.2, random_state = seed)
print('split')
svm_model = svm.SVC(kernel='linear')
print('svm')
pipe = svm_model
pipe.fit(train_data, train_label)
print('fit')
y_pred = pipe.predict(test_data)
f1 = f1_score(test_label, y_pred)
print(f"The f1 of the model is {round(f1,3)}")
