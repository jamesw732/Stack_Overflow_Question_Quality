from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import os

# Train Logistic regression on the training set and measure error on the test set.

datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

X_train = pd.read_csv(os.path.join(datadir, 'train.csv'))
y_train = X_train['score'] > 0
drop = ['score', 'is_answered']
X_train = X_train[X_train.columns.drop(drop)]

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

X_test = pd.read_csv(os.path.join(datadir, 'test.csv'))
y_test = X_test['score'] > 0
X_test = X_test[X_test.columns.drop(drop)]

predict = lr.predict(X_test)
CE = len(np.where(predict != y_test)[0]) / len(y_test)
CS = 1 - CE
print(f"Test Classification error: {CE}")
print(f"Test Classification score: {CS}")