#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:01:13 2017

@author: AlexandraDing
"""

import numpy as np
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss 
from sklearn.metrics import make_scorer
from sklearn.metrics import jaccard_similarity_score as jaccard_score
from sklearn.metrics import classification_report


os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group')

### SUMMARIZE MODEL ACCURACY: 
    # for MULTILABEL DATA, calculates baseline accuracy, hamming loss, f1 score, jaccard similarity, classification report
    # INPUTS:
        # y_prediction: predicted y
        # y_data : ground truth y
    # OUTPUTS:
        # prints accuracy metrics
        # Return 0

def summarize_model_accuracy (y_prediction, y_data):
    # Get basic accuracy: what proportion of labels are correct
    print 'Accuracy:', np.mean(y_prediction == y_data)
    
    # Get Hamming Loss
    print 'Hamming Loss:', hamming_loss(y_data, y_prediction)
    
    # Get f1
    print 'F1 Score:', f1_score(y_data, y_prediction, average = 'weighted')
    
    # get Jaccard Similarity
    print 'Jaccard Similarity:', jaccard_score(y_data, y_prediction)
    
    # Classification report:report recall, precision, f1 ON EACH CLASS (can be used for multilabel case)
    print classification_report(y_data, y_prediction)
    
### Load Dataset
# X: Unprocessed features
# X_std: standardized by Preprocessor
# y: MultiLabel Binarized targets
[X_data, X_data_std, y_data] = pickle.load(open('continuous_features_targets.p', 'rb'))

print 'X_data shape:', X_data.shape
print 'X_data_std shape:', X_data_std.shape
print 'y shape', y_data.shape


data_wd = '/Users/AlexandraDing/Documents/cs109b-best-group/Model_Results/'
[LogReg_cv, y_pred_LogReg] = pickle.load(open(data_wd+'LogReg_grid_results.p', 'rb'))
#print y_pred_LogReg.shape

[DT_grid_cv, y_pred_Decision_Tree] = pickle.load(open(data_wd + 'DecisionTree_grid_results.p', 'rb'))
#print y_pred_Decision_Tree.shape

[rf_grid_cv, y_pred_RF] = pickle.load(open(data_wd + 'RandomForest_tune_maxdepth_hamming_grid_results.p', 'rb'))
#print y_pred_RF.shape


prediction_list = [y_pred_LogReg, y_pred_Decision_Tree, y_pred_RF]

for y_prediction in prediction_list:
    summarize_model_accuracy(y_prediction, y_data)

### Try to visualize CV result
type(LogReg_cv)
LogReg_cv.keys()
plt.semilogx(LogReg_cv['param_estimator__C'], LogReg_cv['mean_test_score'])



