# Parameter grid search with xgboost
# feature engineering is not so useful and the LB is so overfitted/underfitted
# so it is good to trust your CV

# go xgboost, go mxnet, go DMLC! http://dmlc.ml

# Credit to Shize's R code and the python re-implementation

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../dump/train_modified.csv')
test_df = pd.read_csv('../dump/test_modified.csv')

target = 'acc_now_delinq'
IDcol = 'member_id'
train_df[target].value_counts()

features = [x for x in train_df.columns if x not in [target, IDcol]]

xgb_model = xgb.XGBClassifier()

# brute force scan for all parameters, here are the tricks
# usually max_depth is 6,7,8
# learning rate is around 0.05, but small changes may make big diff
# tuning min_child_weight subsample colsample_bytree can have
# much fun of fighting against overfit
# n_estimators is how many round of boosting
# finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread': [-1],  # when use hyperthread, xgboost may become slower
              'objective': ['binary:logistic'],
              'learning_rate': [0.05],  # so called `eta` value
              'max_depth': [9],
              'min_child_weight': [1],
              'gamma': [0],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'n_estimators': [1000],  # number of trees, change it to 1000 for better results
              # 'missing': [-999],
              'scale_pos_weight': [100],  # 正负样本比
              'seed': [27]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   cv=StratifiedKFold(n_splits=5, shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train_df[features], train_df[target])

# trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test_df[features])[:, 1]

sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
