
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


z = '2X'
dts = [f'Forum77data_{z}_train.csv']
bts = [f'Forum77data_{z}_test.csv']

data = pd.read_csv(dts[0])
test = pd.read_csv(bts[0])

train_label = data.pop('Class') # Class
train_data = data.drop('Topic',axis = 1)

test_label = test.pop('Class') # Class
test_data = test.drop('Topic',axis = 1)


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

print(model.feature_importances_)

#X_train_encoded = encode_and_bind(train_data, features_to_encode)

#--------------------------------------------------------


accuracy = accuracy_score(test_label, y_pred)
recall = recall_score(test_label, y_pred)
precision = precision_score(test_label, y_pred)

print(f"The accuracy of the model is {round(accuracy,3)*100} %")
print(f"The recall of the model is {round(recall,3)*100} %")
print(f"The precision of the model is {round(precision,3)*100} %")
print(f"The f1 is {round(f1_score(test_label, y_pred),3)}")




train_probs = model.predict_proba(train_data)[:,1] 
probs = model.predict_proba(test_data)[:, 1]
train_predictions = model.predict(train_data)

print(f'Train ROC AUC Score: {roc_auc_score(train_label, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_label, probs)}')

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


evaluate_model(y_pred,probs,train_predictions,train_probs)

#cm = confusion_matrix(test_label, y_pred)
#plot_confusion_matrix(cm, classes = ['0 - Exploit', '1 - Not Exploit'],
#                      title = 'Exploit Confusion Matrix')