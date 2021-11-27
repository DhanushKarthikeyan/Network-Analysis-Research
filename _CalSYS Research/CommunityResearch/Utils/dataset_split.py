
import pandas

def spltdataset(data, test):

    y_train = data.pop('Class') # Class
    X_train = data.drop('Topic',axis = 1)

    y_test = test.pop('Class') # Class
    X_test = test.drop('Topic',axis = 1)

    return X_test, X_train, y_test, y_train