import xgboost as xgb
from xgboost import cv
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

#creating train set
train = pd.read_csv(os.path.join(datadir, 'train.csv'))
good_score = (train['score'] > 0)*1
drop = ['score', 'is_answered']
train = train[train.columns.drop(drop)]

#creating test set 
test = pd.read_csv(os.path.join(datadir, 'test.csv'))
test_good_score = test['score'] > 0
test = test[test.columns.drop(drop)]

#putting data into optimized DMatrix
data_dmatrix = xgb.DMatrix(data = train, label = good_score)

#setting XGBoost classifier parameters (might want to research how to optimize these)
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'base_score':0.5,
            'booster':'gbtree',
            'colsample_bylevel':1,
            'colsample_bynode':1,
            'colsample_bytree':1,
            'gamma':0,
            'max_delta_step':0,
            'min_child_weight':1,
            'n_jobs':1,
            'random_state':0,
            'reg_alpha':0,
            'reg_lambda':0,
            'scale_pos_weight':1,
            'subsample':1,
            'verbosity':1
        }         

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

print(xgb_cv.head())

#fitting classification model
xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(train, good_score)

y_pred = xgb_clf.predict(test)

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(test_good_score, y_pred)))

xgb.plot_importance(xgb_clf)
plt.show()


