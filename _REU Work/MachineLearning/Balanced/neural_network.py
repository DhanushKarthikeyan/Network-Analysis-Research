from numpy import loadtxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import optimizers

data = pd.read_csv('training_tfidf_description.csv', delimiter=',')
test = pd.read_csv('testing_tfidf_description.csv', delimiter=',')
#print(data['train_label'].dtype)

label_map = {True: 1, False: 0}
data['train_label'] = data['train_label'].map(label_map)
test['train_label'] = test['train_label'].map(label_map)

seed = 50 
#train_data, test_data, train_label, test_label = train_test_split(X,train_label,test_size=0.1, random_state = seed)
train_label = data.pop('train_label')
train_data = data.drop('assign_date',axis = 1)
train_data = train_data.drop('cve_publish_date',axis = 1)
train_data = train_data.drop('cve_id',axis = 1)
train_data = train_data.drop('cwe_id',axis = 1)
train_data = train_data.drop('cvss2_basescore',axis = 1)
train_data = train_data.drop('cvss2_exploitabilityscore',axis = 1)
train_data = train_data.drop('cvss2_impactscore',axis = 1)
train_data = train_data.drop('cvss2_obtainallprivilege',axis = 1)
train_data = train_data.drop('cvss2_obtainotherprivilege',axis = 1)
train_data = train_data.drop('cvss2_obtainuserprivilege',axis = 1)
train_data = train_data.drop('cvss2_severity',axis = 1)
train_data = train_data.drop('cvss2_userinteractionrequired',axis = 1)
train_data = train_data.drop('cve_description',axis = 1)

test_label = test.pop('train_label')
test_data = test.drop('assign_date',axis = 1)
test_data = test_data.drop('cve_publish_date',axis = 1)
test_data = test_data.drop('cve_id',axis = 1)
test_data = test_data.drop('cwe_id',axis = 1)
test_data = test_data.drop('cvss2_basescore',axis = 1)
test_data = test_data.drop('cvss2_exploitabilityscore',axis = 1)
test_data = test_data.drop('cvss2_impactscore',axis = 1)
test_data = test_data.drop('cvss2_obtainallprivilege',axis = 1)
test_data = test_data.drop('cvss2_obtainotherprivilege',axis = 1)
test_data = test_data.drop('cvss2_obtainuserprivilege',axis = 1)
test_data = test_data.drop('cvss2_severity',axis = 1)
test_data = test_data.drop('cvss2_userinteractionrequired',axis = 1)
test_data = test_data.drop('cve_description',axis = 1)

features_to_encode = train_data.columns[train_data.dtypes==object].tolist()  

train_label = train_label.values
train_data = train_data.values
test_label = test_label.values
test_data = test_data.values

train_data = train_data.astype(str)
test_data = test_data.astype(str)
train_label = train_label.reshape((len(train_label), 1))
test_label = test_label.reshape((len(test_label), 1))

#Train (316, 17) (316, 1)
#Test (1334, 17) (1334, 1)

# prepare input data
def prepare_inputs(X_train, X_test):
	ohe = OneHotEncoder(handle_unknown='ignore')
	ohe.fit(X_train)
    
	X_train_enc = ohe.transform(X_train)
	X_test_enc = ohe.transform(X_test)
	return X_train_enc, X_test_enc

def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# prepare input data
train_data_enc, test_data_enc = prepare_inputs(train_data, test_data)
# prepare output data
train_label_enc, test_label_enc = prepare_targets(train_label, test_label)

model = Sequential()
model.add(Dense(10, input_dim=train_data_enc.shape[1], activation='relu'))
model.add(Dense(train_data_enc.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

opt = optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt,  metrics=['accuracy',f1_m,precision_m, recall_m])

# fit the keras model on the dataset
model.fit(train_data_enc, train_label_enc, epochs=500, batch_size=100, verbose=2)

loss, accuracy, f1_score, precision, recall = model.evaluate(test_data_enc, test_label_enc, verbose=0)

print('Accuracy: %.2f' % (accuracy))
print('loss: %.2f' % (loss))
print('f1_score: %.2f' % (f1_score))
print('precision: %.2f' % (precision))
print('recall: %.2f' % (recall))
