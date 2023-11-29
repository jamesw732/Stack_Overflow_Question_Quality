import xgboost as xgb
import pandas as pd
import numpy as np
import os

datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

train = pd.read_csv(os.path.join(datadir, 'train.csv'))
good_score = (train['score'] > 0)*1
drop = ['score', 'is_answered']
train = train[train.columns.drop(drop)]

data_dmatrix = xgb.DMatrix(data = train, label = good_score)

params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }         

xgb_clf = xgb.XGBClassifier(**params)

xgb_clf.fit(train, good_score)

print(xgb_clf)

