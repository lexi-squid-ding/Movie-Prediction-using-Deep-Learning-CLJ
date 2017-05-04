#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:54:48 2017

@author: AlexandraDing
"""
#########################################
#### SVM Models Only #### 
# Script runs Linear, Polynomial and RBF SVM CV for multi-label classification
#########################################

import numpy as np
import pandas as pd
import pickle
import os

### Cross Validation and Model Selection metrics

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

from sklearn.metrics import hamming_loss 
from sklearn.metrics import make_scorer

### Classification

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


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


######      TUNE SVM MODELS       ######

### SVM: Linear
# Tune: C (regularization parameter)
SVM_Linear_Model = OneVsRestClassifier(SVC(kernel = 'linear'))

# Creae Grid Search on parameters with hamming loss
SVM_Linear_grid = GridSearchCV(SVM_Linear_Model,
                    param_grid = {'estimator__C': np.logspace(-5, 5, 11)}, 
                                  scoring = hamming_scorer)
SVM_Linear_grid.fit(X_data_std, y_data)
SVM_Linear_grid.cv_results_['mean_test_score']

print 'Best score Linear SVM:', SVM_Linear_grid.best_score_


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
