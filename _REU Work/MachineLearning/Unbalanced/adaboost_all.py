import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('labeled_all_dw.csv', delimiter=',')

label_map = {'t': 1, 'f': 0}
data['train_label'] = data['train_label'].map(label_map)

label = data.pop('train_label')
data = data.drop('assign_date',axis = 1)
data= data.drop('cve_publish_date',axis = 1)
data = data.drop('cve_id',axis = 1)
data = data.drop('cwe_id',axis = 1)
data = data.drop('cvss2_basescore',axis = 1)
data = data.drop('cvss2_exploitabilityscore',axis = 1)
data = data.drop('cvss2_impactscore',axis = 1)
data = data.drop('cvss2_obtainallprivilege',axis = 1)
data = data.drop('cvss2_obtainotherprivilege',axis = 1)
data = data.drop('cvss2_obtainuserprivilege',axis = 1)
data = data.drop('cvss2_severity',axis = 1)
data = data.drop('cvss2_userinteractionrequired',axis = 1)


seed = 50 
train_data, test_data, train_label, test_label = train_test_split(data,label,test_size=0.333, random_state = seed)

features_to_encode = train_data.columns[train_data.dtypes==object].tolist()  

col_trans = make_column_transformer(
                        (OneHotEncoder(handle_unknown='ignore'),features_to_encode),
                        remainder = "passthrough"
                        )

#plum
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=False,
                      oob_score=False,
                      class_weight='balanced_subsample',
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')

model = AdaBoostClassifier(random_state=50, base_estimator= rf_classifier, n_estimators=100, learning_rate= 0.001)

pipe = make_pipeline(col_trans, model)
pipe.fit(train_data, train_label)

y_pred = pipe.predict(test_data)

def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)

X_train_encoded = encode_and_bind(train_data, features_to_encode)

feature_importances = list(zip(X_train_encoded, model.feature_importances_))
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
#[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]


'''# Plot the top 25 feature importance
num_features = 15
feature_names = [i[0] for i in feature_importances_ranked[:num_features]]
y_ticks = np.arange(0, len(feature_names))
x_axis = [i[1] for i in feature_importances_ranked[:num_features]]
plt.figure(figsize = (10, 7))
plt.barh(feature_names, x_axis)   #horizontal barplot
plt.title(f'Adaboost with RF Feature Importance (Top {num_features})',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 18})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()'''

#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

accuracy = accuracy_score(test_label, y_pred)
recall = recall_score(test_label, y_pred)
precision = precision_score(test_label, y_pred)
f1 = f1_score(test_label, y_pred)

print("Adaboost with RF")
print(f"The accuracy of the model is {round(accuracy,3)*100} %")
print(f"The recall of the model is {round(recall,3)*100} %")
print(f"The precision of the model is {round(precision,3)*100} %")
print(f"The f1 of the model is {round(f1,3)}")

train_probs = pipe.predict_proba(train_data)[:,1] 
probs = pipe.predict_proba(test_data)[:, 1]
train_predictions = pipe.predict(train_data)

print(f'Train ROC AUC Score: {roc_auc_score(train_label, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_label, probs)}')
