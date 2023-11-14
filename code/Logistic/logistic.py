from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pathlib
import os

# Perform logistic regression with two definitions for "good question":
# 1. score >= 0
# 2. is_answered
# Don't use metadata features for this, as we are interested in interpretability.

dir = os.path.join(pathlib.Path(__file__).parent.resolve())

# Score:
train = pd.read_csv(os.path.join(dir, 'score_train.csv'))
good_score = train['good_score']
X_train = train[train.columns.drop(['good_score'])]
test = pd.read_csv(os.path.join(dir, 'score_test.csv'))
good_score_test = test['good_score']
X_test = test[test.columns.drop(['good_score'])]

lr = LogisticRegression().fit(X_train, good_score)

predict = lr.predict(X_test)
error = len(np.where(predict != good_score_test)[0]) / len(good_score_test)
print(f"Classification error using score as definition of good question: {error * 100}%")


train = pd.read_csv(os.path.join(dir, 'answered_train.csv'))
answered = train['is_answered']
X_train = train[train.columns.drop(['is_answered'])]
test = pd.read_csv(os.path.join(dir, 'answered_train.csv'))
answered_test = test['is_answered']
X_test = test[test.columns.drop(['is_answered'])]

lr = LogisticRegression(max_iter=1000).fit(X_train, answered)

predict = lr.predict(X_test)
error = len(np.where(predict != answered_test)[0]) / len(answered_test)
print(f"Classification error using answered/not answered as definition of good question: {error * 100}%")