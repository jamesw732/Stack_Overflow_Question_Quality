from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (cross_val_score, StratifiedKFold)
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Train Logistic regression on the training set and measure error on the test set.

def cv(X_train, y_train, lr):
    kfold = StratifiedKFold(n_splits=5)
    return np.mean(cross_val_score(lr, X_train, y_train, cv=kfold))

datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../data'))

X_train = pd.read_csv(os.path.join(datadir, 'train.csv'))
y_train = X_train['score'] > 0
drop = ['score', 'is_answered']
X_train = X_train[X_train.columns.drop(drop)]

rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

cv_score = cv(X_train, y_train, lr)
print(f"Logistic Regression Cross Validation Score: {cv_score}")

# Uncomment this if lr gets best cross validation score

# X_test = pd.read_csv(os.path.join(datadir, 'test.csv'))
# y_test = X_test['score'] > 0
# X_test = X_test[X_test.columns.drop(drop)]

# predict = lr.predict(X_test)
# CE = len(np.where(predict != y_test)[0]) / len(y_test)
# CS = 1 - CE
# print(f"Test Classification error: {CE}")
# print(f"Test Classification score: {CS}")



"""Feature Importance"""

#Permutation importance:
# importance = permutation_importance(lr, pd.concat([X_train, X_test]), 
#                                     pd.concat([y_train, y_test]), n_repeats=30,
#                                     random_state=0)

# print([(name, val) for val, name in sorted(zip(importance.importances_mean, X_train.columns), reverse=True)])

# Normalized coefficients:
norm_X_train = (X_train - X_train.mean()) / X_train.std()
lr_norm = LogisticRegression(max_iter=1000)
lr_norm.fit(X_train, y_train)
importance = [[name, float(val)] for val, name in sorted(zip(lr_norm.coef_[0], X_train.columns), key=lambda x: abs(x[0]))]
labels = [row[0] for row in importance]
widths = [row[1] for row in importance]
# importance[1] = [np.format_float_scientific(x, precision=3) for x in importance[1]]

fig, ax = plt.subplots()
ax.barh(labels, widths, align='center', height=0.2)
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha = 0.2)
ax.set_ylabel("Features")
ax.set_xlabel("Normalized Logistic Regression Coefficients")
plt.subplots_adjust(left=0.25)
# plt.show()
savedir = os.path.abspath(os.path.join(os.path.realpath(__file__), 
                        '../../../report/figures/normalized_logistic_coef.png'))
plt.savefig(savedir, bbox_inches='tight')