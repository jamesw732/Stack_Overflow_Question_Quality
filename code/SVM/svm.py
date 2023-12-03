from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os

# Train SVC on training set and measure error on test set.
# Cross validate kernel.

def cv(X_train, y_train, svc):
    cv = cross_validate(svc, X_train, y_train, scoring='accuracy', cv=5)
    return np.mean(cv['test_score'])

if __name__ == "__main__":
    datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

    X_train = pd.read_csv(os.path.join(datadir, 'train.csv'))
    y_train = X_train['score'] > 0
    drop = ['score', 'is_answered']
    X_train = X_train[X_train.columns.drop(drop)]

    # PCA:
    # pca = PCA(n_components=2)
    # pca.fit(X_train)
    # X_train = X_train @ pca.components_.T

    # Scale:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    svm_linear = SVC(kernel="linear", probability=False, max_iter=1000000)
    svm_poly = SVC(kernel="poly", probability=False)
    svm_rbf = SVC(probability=False)

    print("Starting Cross Validation")
    cv_linear = cross_validate(svm_linear, X_train, y_train, scoring='accuracy', cv=7)
    print(np.mean(cv_linear['test_score']))
    cv_poly = cross_validate(svm_poly, X_train, y_train, scoring='accuracy', cv=7)
    print(np.mean(cv_poly['test_score']))
    cv_rbf = cross_validate(svm_rbf, X_train, y_train, scoring='accuracy', cv=7)
    print(np.mean(cv_rbf['test_score']))

    svm_linear.fit(X_train, y_train)
    X_test = pd.read_csv(os.path.join(datadir, 'test.csv'))
    y_test = X_test['score'] > 0
    X_test = X_test[X_test.columns.drop(drop)]
    X_test = sc.transform(X_test)

    predict = svm_linear.predict(X_test)
    CE = len(np.where(predict != y_test)[0]) / len(y_test)
    CS = 1 - CE
    print(f"Linear Kernel Test Classification error: {CE}")
    print(f"Linear Kernel Test Classification score: {CS}")