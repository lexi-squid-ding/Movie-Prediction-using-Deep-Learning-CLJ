#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:04:43 2017

@author: AlexandraDing
"""

#########################################
#### Logistic Regression and Random Forest Models #####
#########################################
import numpy as np
import pandas as pd
import pickle
import os


### Cross Validation and Model Selection metrics

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss 
from sklearn.metrics import make_scorer

# Preprocessing
import sklearn.preprocessing as Preprocessing
from sklearn.preprocessing import StandardScaler as Standardize
from sklearn.preprocessing import MultiLabelBinarizer

### Classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.linear_model import LogisticRegression as Log_Reg
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Dimensionality Reduction
from sklearn.decomposition import PCA

### THINGS TO CHANGE
os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group')
# MAY ALSO WANT TO CHANGE N_JOBS ARGUMENT IN GRID


### Load Dataset
# X: Unprocessed features
# X_std: standardized by Preprocessor
# y: MultiLabel Binarized targets
[X_data, X_data_std, y_data] = pickle.load(open('continuous_features_targets.p', 'rb'))

print 'X_data shape:', X_data.shape
print 'X_data_std shape:', X_data_std.shape
print 'y shape', y_data.shape


######      CREATE SCORING METRIC        ######
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html

# Hamming Loss: want to MINIMIZE LOSS 
hamming_scorer = make_scorer(hamming_loss, greater_is_better = False)


### Logistic Regression
import time 
tic = time.time()
# Try one instance of LogisticRegression
LR_test = OneVsRestClassifier(Log_Reg(penalty = 'l2', class_weight = 'balanced'))
LR_test.fit(X_data_std, y_data)
toc = time.time()
duration = toc - tic
print duration

# l2 regularization (tuned) + balanced class weights (i.e. based on label frequency)
tic = time.time()
LogReg_Model = OneVsRestClassifier(Log_Reg(penalty = 'l2', class_weight = 'balanced'))

LogReg_grid = GridSearchCV(LogReg_Model, 
                           param_grid={'estimator__C': np.logspace(-5, 15, 20)}, 
                                       scoring= hamming_scorer,
                                       n_jobs = 5)
LogReg_grid.fit(X_data_std, y_data)
LogReg_grid.cv_results_['mean_test_score']
toc = time.time()
duration = toc - tic
print duration
# Best score has haming loss of 0.354 (reported as -0.354 because we are maximizing the function)
print 'Best score LogReg:', LogReg_grid.best_score_

# Fit best model on data, predict
y_pred_LogReg = cross_val_predict(LogReg_grid.best_estimator_, X_data_std, y_data)

# Dump CV results AND predictions from best model
pickle.dump([LogReg_grid.cv_results_, y_pred_LogReg], open('LogReg_grid_results.p', 'wb'))




### Single Decision Tree
tic = time.time()

DecisionTree_Model = OneVsRestClassifier(tree.DecisionTreeClassifier(criterion='gini'))
DT_grid = GridSearchCV(DecisionTree_Model, 
                    param_grid = {'estimator__max_depth': range(1,10)},
                                  scoring = hamming_scorer)
DT_grid.fit(X_data_std, y_data)
DT_grid.cv_results_['mean_test_score']
print 'Best score single DT:', DT_grid.best_score_

y_pred_Decision_Tree = cross_val_predict(DT_grid.best_estimator_, X_data_std, y_data)
np.mean(y_pred_Decision_Tree == y_data)

pickle.dump([DT_grid.cv_results_, y_pred_Decision_Tree], open('DecisionTree_grid_results.p', 'wb'))

print time.time()- tic


### Random Forest: Classical Random Forest
# Tune: max_depth

#tic = time.time()
#
#RandomForest_Model = OneVsRestClassifier(RandomForest())
#rf_grid = GridSearchCV(RandomForest_Model,
#                       param_grid = {'estimator__max_depth': [10, 20, 30] })
#
#rf_grid.fit(X_data_std, y_data)
#rf_grid.cv_results_['mean_test_score']
#print 'Best score in RF tuning max_depth:', rf_grid.best_score_
#y_pred_RF = cross_val_predict(rf_grid.best_estimator_, X_data_std, y_data)
#
#pickle.dump([rf_grid.cv_results_, y_pred_RF], open('RandomForest_tune_maxdepth_grid_results.p', 'wb'))
#
#print time.time()- tic

# Larger range of max_depth tuned with Hamming loss
tic = time.time()

RandomForest_Model = OneVsRestClassifier(RandomForest())
rf_grid = GridSearchCV(RandomForest_Model,
                       param_grid = {'estimator__max_depth': 10*np.linspace(1,7, 7) },
scoring = hamming_scorer)

rf_grid.fit(X_data_std, y_data)
rf_grid.cv_results_['mean_test_score']
print 'Best score in RF tuning max_depth:', rf_grid.best_score_
y_pred_RF = cross_val_predict(rf_grid.best_estimator_, X_data_std, y_data)

pickle.dump([rf_grid.cv_results_, y_pred_RF], open('RandomForest_tune_maxdepth_hamming_grid_results.p', 'wb'))

print time.time()- tic





#### SPECIAL FLAVORS OF DECISION TREES
### Random Forest, Ada boosted: 100 trees
# NOTE: Need to tune n_estimators (n_trees)

#Ada_Model = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100))
#Ada_Model.fit(X_data_std, y_data)
#scores = cross_val_score(Ada_Model,X_data_std, y_data, scoring = hamming_scorer)

#from sklearn.datasets import load_iris
#iris = load_iris()
clf = OneVsRestClassifier(AdaBoostClassifier())
ada_grid = GridSearchCV(clf,
                       param_grid = {'estimator__n_estimators': np.logspace(1,3,6).astype(int) },
                                     scoring = hamming_scorer)
#ada_grid.fit(iris.data, iris.target)
ada_grid.fit(X_data_std, y_data)
ada_grid.cv_results_['mean_test_score']
print 'Best score in RF tuning max_depth:', ada_grid.best_score_

#scores = cross_val_score(ada_grid.best_estimator_, X_data_std, y_data, scoring = hamming_scorer)
y_pred_ada = cross_val_predict(ada_grid.best_estimator_, X_data_std, y_data)
pickle.dump([ada_grid.cv_results_, y_pred_ada], open('Adaboost_grid_results.p', 'wb'))


### Random Forest, Gradient boosting
GBRT_Model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100))
scores = cross_val_score(GBRT_Model, X_data_std, y_data, scoring = hamming_scorer)





