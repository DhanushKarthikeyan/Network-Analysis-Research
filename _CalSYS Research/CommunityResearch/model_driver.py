from numpy.lib.function_base import rot90
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

def lazy(X_test, X_train, y_test, y_train, forum, m):
    # fit all models
    clf = LazyClassifier(verbose = 0, predictions=True)
    # models is a dataframe, predictions are predictions
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)    
    
    fig = plt.figure()
    mdls = models.index.tolist()
    f1_score = models.loc[:,"F1 Score"].tolist()
    thr = 0.6
    mdls, f1_score = zip(*[(a, b) for a, b in zip(mdls, f1_score) if b>thr]) #only above 70% f1 score

    plt.bar(x = mdls, height = f1_score)
    plt.xticks(rotation = 70)
    plt.tick_params(axis='x', which='major', labelsize=7)
    plt.title(f'Performance of Different Classifiers in Forum{forum} {m} above {thr}', fontsize = 16, fontweight = 'bold')
    #plt.show()

    fig.savefig(f"F{forum}_{m}X_new", dpi = 500, bbox_inches='tight')
    return 0

def RF_ADA(X_test, X_train, y_test, y_train, forum, m):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    rf_classifier = RandomForestClassifier(
    n_estimators=900,
    random_state=0,
    criterion ='gini'
    )
    model = AdaBoostClassifier(random_state=50, base_estimator= rf_classifier, n_estimators=100, learning_rate= 0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f'\n RF_ADA in Forum {forum} with {m}X')
    #print(f"Accuracy: {round(accuracy,3)*100}% F1: {round(f1_score(y_test, y_pred),3)}")
    #print(f"The accuracy of the model is {round(accuracy,3)*100} %")
    #print(f"The recall of the model is {round(recall,3)*100} %")
    #print(f"The precision of the model is {round(precision,3)*100} %")
    #print(f"The f1 is {round(f1_score(y_test, y_pred),3)}")
    return round(accuracy,3)*100, round(f1_score(y_test, y_pred),3)

def train_models(X_test, X_train, y_test, y_train, forum, m, models):
    if 'lazy' in models:
        lazy(X_test, X_train, y_test, y_train, forum, m)
    
    if 'RF_ADA' in models:
        acc, f1 = RF_ADA(X_test, X_train, y_test, y_train, forum, m)
        return acc, f1
