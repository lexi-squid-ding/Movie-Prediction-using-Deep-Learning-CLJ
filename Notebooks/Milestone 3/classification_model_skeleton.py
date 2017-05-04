#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:01:23 2017

@author: AlexandraDing
"""

import numpy as np
import pandas as pd
import pickle

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Dimensionality Reduction
from sklearn.decomposition import PCA


import os
os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group')

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


######     TEST MODEL OVERFITTING        ######

# initiate the model with linear SVC
OVR_svc = OneVsRestClassifier(LinearSVC())

# fit the model
OVR_svc.fit(X_data_std, y_data)

# Predict on whole dataset (overfit)
Y_pred = OVR_svc.predict(X_data_std)
print 'Accuracy (overfit)', np.mean(Y_pred == y_data)
print 'Hamming Loss (overfit)', hamming_loss(y_data, Y_pred)
#Accuracy (overfit) 0.914051034023
#Hamming Loss (overfit) 0.0859489659773


############### MODEL FITTING ############### 
# From: http://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
# When you use nested estimators with grid search you can scope the parameters with __ as a separator.
# GridSearchCV: defaults to 3-fold cross validation

### Logistic Regression
# l2 regularization (tuned) + balanced class weights (i.e. based on label frequency)
LogReg_Model = OneVsRestClassifier(Log_Reg(penalty = 'l2', class_weight = 'balanced'))

LogReg_grid = GridSearchCV(LogReg_Model, 
                           param_grid={'estimator__C': np.logspace(-5, 15, 20)}, 
                                       scoring= hamming_scorer,
                                       n_jobs = 5)
LogReg_grid.fit(X_data_std, y_data)
LogReg_grid.cv_results_['mean_test_score']

# Best score has haming loss of 0.354 (reported as -0.354 because we are maximizing the function)
print 'Best score LogReg:', LogReg_grid.best_score_

# Fit best model on data, predict
y_pred_LogReg = cross_val_predict(LogReg_grid.best_estimator_, X_data_std, y_data)

# Dump CV results AND predictions from best model
pickle.dump([LogReg_grid.cv_results_, y_pred_LogReg], open('LogReg_grid_results.p', 'wb'))



### SVM: Linear
# Tune: C (regularization parameter)
SVM_Linear_Model = OneVsRestClassifier(SVC(kernel = 'linear'))
SVM_Linear_grid = GridSearchCV(SVM_Linear_Model,
                    param_grid = {'estimator__C': np.logspace(-5, 5, 11)}, 
                                  scoring = hamming_scorer,
                                  n_jobs = 5)
SVM_Linear_grid.fit(X_data_std, y_data)
SVM_Linear_grid.cv_results_['mean_test_score']

print 'Best score Linear SVM:', SVM_Linear_grid.best_score_


y_pred_SVM_Linear = cross_val_predict(SVM_Linear_grid.best_estimator_, X_data_std, y_data)

# Fit best model on data, predict
y_pred_svm_linear = cross_val_predict(SVM_Linear_grid.best_estimator_, X_data_std, y_data)

# Dump CV results AND predictions from best model
pickle.dump([SVM_Linear_grid.cv_results_, y_pred_svm_linear], open('svm_linear_grid_results.p', 'wb'))


### SVM: Polynomial
# tune: C, degree
SVM_Poly_Model = OneVsRestClassifier(SVC(kernel = 'poly'))

svm_poly_parameters = {
    "estimator__C": np.logspace(-5, 5, 11),
    "estimator__degree": range(1,5),
}

svm_poly_grid = GridSearchCV(SVM_Poly_Model, 
                    param_grid = svm_poly_parameters,
                    scoring = hamming_scorer)
svm_poly_grid.fit(X_data_std, y_data)


print 'Best score Poly SVM:', svm_poly_grid.best_score_

# Fit best model on data, predict
y_pred_svm_poly = cross_val_predict(svm_poly_grid.best_estimator_, X_data_std, y_data)

# Dump CV results AND predictions from best model
pickle.dump([svm_poly_grid.cv_results_, y_pred_svm_poly], open('svm_poly_grid_results.p', 'wb'))


### SVM: RBF kernel
# Tune: C, gamma
SVM_RBF_Model = OneVsRestClassifier(SVC(kernel = 'rbf'))

svm_rbf_parameters = {
    "estimator__C": np.logspace(-5, 5, 11),
    "estimator__gamma": np.logspace(-5, 5, 11)
}

svm_rbf_grid = GridSearchCV(SVM_RBF_Model, 
                    param_grid= svm_rbf_parameters,
                    scoring = hamming_scorer)
svm_rbf_grid.fit(X_data_std, y_data)

print 'Best score RBF SVM:', svm_rbf_grid.best_score_

# Fit best model on data, predict
y_pred_svm_rbf = cross_val_predict(svm_rbf_grid.best_estimator_, X_data_std, y_data)

# Dump CV results AND predictions from best model
pickle.dump([svm_rbf_grid.cv_results_, y_pred_svm_rbf], open('svm_rbf_grid_results.p', 'wb'))



### Single Decision Tree
DecisionTree_Model = OneVsRestClassifier(tree.DecisionTreeClassifier(criterion='gini'))
grid = GridSearchCV(DecisionTree_Model, 
                    param_grid = {'estimator__max_depth': range(1,10)},
                                  scoring = hamming_scorer)
grid.fit(X_data_std, y_data)
grid.cv_results_['mean_test_score']
grid.best_score_


### Random Forest: Classical Random Forest
# Tune: max_depth, min_samples_leaf 

RandomForest_Model = OneVsRestClassifier(RandomForest())
rf_grid = GridSearchCV(RandomForest_Model,
                       param_grid = {'estimator__max_depth': [10, 20, 30] })

rf_grid.fit(X_data_std, y_data)
rf_grid.cv_results_['mean_test_score']
rf_grid.best_score_


### Random Forest, Ada boosted: 100 trees
# NOTE: Need to tune learning rate
Ada_Model = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100))
Ada_Model.fit(X_data_std, y_data)
scores = cross_val_score(Ada_Model,X_data_std, y_data, scoring = hamming_scorer)


### Random Forest, Gradient boosting
GBRT_Model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100))
scores = cross_val_score(GBRT_Model, X_data_std, y_data, scoring = hamming_scorer)


### LDA: Linear Discriminant Analysis, tune prior, with default Ledoit-Wolf shrinkage
# NOTE: Need to write code for prior! 
LDA_Model = OneVsRestClassifier(LDA(shrinkage = 'auto'))


### QDA: Quadratic Discriminant Analyis
# NOTE: Need to write code for prior! 
QDA_Model = OneVsRestClassifier(QDA(shrinkage = 'auto'))


###### Dimensionality Reduction Algorithms ###### 

# Consider running dimensionality reduction beforehand
 
#model = tree.DecisionTreeClassifier(criterion='gini', max_depth = 2)
#
## Split data into folds
#n_folds = 3
#
#kfold_cv = KFold(n_splits = n_folds, shuffle = True)
#score = np.zeros((n_folds))
#
#i = 0
#for train, test in kfold_cv.split(x_std, Y):
#    model.fit(x_std[train], Y[train])
#    
#    # Using f1 score
#    score[i] = f1_score(Y[test], model.predict(x_std[test]), average= 'macro')
#    i = i+1

#X_train, X_test, y_train, y_test = train_test_split(x_std, Y, test_size=0.33)
##model = OneVsRestClassifier(SVC(kernel='poly'))
#
#model = tree.DecisionTreeClassifier(criterion='gini', max_depth = 2)
#model.fit(X_train, y_train)
#np.mean(y_train == model.predict(X_test))  # predict on a new X

