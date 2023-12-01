from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import os

# Train Logistic regression on the training set and measure error on the test set.

datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

# Score:
train = pd.read_csv(os.path.join(datadir, 'train.csv'))
good_score = train['score'] > 0
drop = ['score', 'is_answered']
train = train[train.columns.drop(drop)]

lr = LogisticRegression(max_iter=1000)

test = pd.read_csv(os.path.join(datadir, 'test.csv'))
test_good_score = test['score'] > 0
test = test[test.columns.drop(drop)]
lr.fit(train, good_score)
predict = lr.predict(test)
CE = len(np.where(predict != test_good_score)[0]) / len(test_good_score)
CS = 1 - CE
print(f"Test Classification error: {CE}")
print(f"Test Classification score: {CS}")