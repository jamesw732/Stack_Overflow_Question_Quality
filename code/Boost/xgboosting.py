import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
            'learning_rate': 0.1,
            'base_score':0.5,
            'booster':'gbtree',
            'colsample_bylevel':1,
            'colsample_bynode':1,
            'colsample_bytree':1,
            'gamma':0.2,
            'max_delta_step':0,
            'min_child_weight':6,
            'n_jobs':1,
            'random_state':0,
            'reg_alpha':0,
            'reg_lambda':0,
            'scale_pos_weight':1,
            'subsample':1,
            'verbosity':1,
            'eta':0.01
        }         

#xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=10, metrics='error', as_pandas=True, seed=43)

#print(xgb_cv)

#fitting classification model
xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(train, good_score)

y_pred = xgb_clf.predict(test)

kfold = StratifiedKFold(n_splits=5)
results = cross_val_score(xgb_clf, test, test_good_score, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(test_good_score, y_pred)))

xgb.plot_importance(xgb_clf,importance_type='gain')


for s in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[s].set_visible(False)
plt.grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha = 0.2)
plt.subplots_adjust(left=0.26)

savedir = os.path.abspath(os.path.join(os.path.realpath(__file__), 
                        '../../../report/figures/xgboost_f_scores.png'))
plt.savefig(savedir, bbox_inches='tight')
# plt.show()


# test_params = {
#  'max_depth':[4,8,12],
#  'eta':[0.01, 0.05, 0.1],
#  'subsample':[0,1,0.2,0.3],
#  'min_child_weight':[4,5,6],
#  'gamma':[0,0.1,0.2]
#  }

# model2 = GridSearchCV(estimator = xgb_clf,param_grid = test_params)
# model2.fit(train,good_score)
# print(model2.best_params_)

#original accuracy score:0.9041
#with eta = 0.01, gamma = 0.02, max_depth = 12, min_child_weight = 6, subsample = 1, score:0.9069

#print(sum(good_score))
#print(good_score.shape)

#1937 good Q's
#2127 - 1937 Bad Q's so use Stratified KFold