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
from sklearn import svm

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

col_trans = make_column_transformer(
                        (OneHotEncoder(handle_unknown='ignore'),features_to_encode),
                        remainder = "passthrough"
                        )

svm_model = svm.SVC(kernel='linear')


pipe = make_pipeline(col_trans, svm_model)
pipe.fit(train_data, train_label)

y_pred = pipe.predict(test_data)

def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)

'''X_train_encoded = encode_and_bind(train_data, features_to_encode)

feature_importances = list(zip(X_train_encoded, rf_classifier.feature_importances_))
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# Print out the feature and importances
#[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]


# Plot the top 25 feature importance
num_features = 15
feature_names = [i[0] for i in feature_importances_ranked[:num_features]]
y_ticks = np.arange(0, len(feature_names))
x_axis = [i[1] for i in feature_importances_ranked[:num_features]]
plt.figure(figsize = (10, 7))
plt.barh(feature_names, x_axis)   #horizontal barplot
plt.title(f'Random Forest Feature Importance (Top {num_features})',
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

print("SVM")
print(f"The accuracy of the model is {round(accuracy,3)*100} %")
print(f"The recall of the model is {round(recall,3)*100} %")
print(f"The precision of the model is {round(precision,3)*100} %")
print(f"The f1 of the model is {round(f1,3)}")


def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(test_label,
                    [1 for _ in range(len(test_label))])
    baseline['precision'] = precision_score(test_label,
                    [1 for _ in range(len(test_label))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(test_label, y_pred)
    results['precision'] = precision_score(test_label, y_pred)
    results['roc'] = roc_auc_score(test_label, probs)
    train_results = {}
    train_results['recall'] = recall_score(train_label,       train_predictions)
    train_results['precision'] = precision_score(train_label, train_predictions)
    train_results['roc'] = roc_auc_score(train_label, train_probs)
    for metric in ['recall', 'precision', 'roc']: 
          print(f'{metric.capitalize()}\
                 Baseline: {round(baseline[metric], 2)} \
                 Test: {round(results[metric], 2)} \
                 Train: {round(train_results[metric], 2)}')
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_label, [1 for _ in range(len(test_label))])
    model_fpr, model_tpr, _ = roc_curve(test_label, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 12)
    plt.colorbar(aspect=5)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 8)
    plt.yticks(tick_marks, classes, size = 8)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 10,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 10)
    plt.xlabel('Predicted label', size = 10)
    plt.show()


#evaluate_model(y_pred,probs,train_predictions,train_probs)

#cm = confusion_matrix(test_label, y_pred)
#plot_confusion_matrix(cm, classes = ['0 - Exploit', '1 - Not Exploit'],
#                      title = 'Exploit Confusion Matrix')
