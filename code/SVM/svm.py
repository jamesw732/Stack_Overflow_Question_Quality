from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import os

# Train SVC on training set and measure error on test set.
# Cross validate kernel.

def cv(X_train, y_train, svc):
    kfold = StratifiedKFold(n_splits=5)
    return np.mean(cross_val_score(svc, X_train, y_train, cv=kfold))

if __name__ == "__main__":
    datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

    X_train = pd.read_csv(os.path.join(datadir, 'train.csv'))
    y_train = X_train['score'] > 0
    drop = ['score', 'is_answered']
    X_train = X_train[X_train.columns.drop(drop)]
    # rus = RandomUnderSampler(random_state=0)
    # X_train, y_train = rus.fit_resample(X_train, y_train)

    # PCA:
    # pca = PCA(n_components=2)
    # pca.fit(X_train)
    # X_train = X_train @ pca.components_.T

    # Scale:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    svm_linear = SVC(kernel="linear", probability=False)
    svm_poly = SVC(kernel="poly", probability=False)
    svm_rbf = SVC(kernel="rbf", probability=False)

    linear_cv_score = cv(X_train, y_train, svm_linear)
    print(f"Linear SVM Cross Validation Score: {linear_cv_score}")
    poly_cv_score = cv(X_train, y_train, svm_poly)
    print(f"Polynomial Cross Validation Score: {poly_cv_score}")
    rbf_cv_score = cv(X_train, y_train, svm_rbf)    
    print(f"RBF Cross Validation Score: {rbf_cv_score}")

    # Uncomment this if SVM gets best cross validation score, and change to the correct one

    # svm_linear.fit(X_train, y_train)
    # X_test = pd.read_csv(os.path.join(datadir, 'test.csv'))
    # y_test = X_test['score'] > 0
    # X_test = X_test[X_test.columns.drop(drop)]
    # X_test = sc.transform(X_test)

    # predict = svm_linear.predict(X_test)
    # CE = len(np.where(predict != y_test)[0]) / len(y_test)
    # CS = 1 - CE
    # print(f"Linear Kernel Test Classification error: {CE}")
    # print(f"Linear Kernel Test Classification score: {CS}")