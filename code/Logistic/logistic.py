from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import pathlib
import os

# Perform logistic regression with two definitions for "good question":
# 1. score > 0 (bad means score < 0, so drop score == 0)
# 2. is_answered
# Goal of this is to decide which definition to use for SVM/Trees. Metadata features seem bad to include here. Source?

dir = os.path.join(pathlib.Path(__file__).parent.resolve())
datadir = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../../data'))

# Score:
train = pd.read_csv(os.path.join(datadir, 'train.csv'))
good_score = train['score'] > 0
is_answered = train['is_answered']
# Controversial lines:
drop = ['score', 'is_answered']
train = train[train.columns.drop(drop)]

lr = LogisticRegression(max_iter=1000)
cv = cross_validate(lr, train, good_score, scoring='accuracy')
print(f"Mean cross-validation score using positive score as definition of good question: {np.mean(cv['test_score'])}")
cv = cross_validate(lr, train, is_answered, scoring='accuracy')
print(f"Mean cross-validation score using answered/not answered as definition of good question: {np.mean(cv['test_score'])}")