#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:34:53 2018

@author: sadams
Cross validation procedures
"""

from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import xgboost as xgb

def cross_val_models(models, X, y, use_cv=5, params=defaultdict(dict), metric = 'roc_auc', verbose = False):
    defaultdict
    '''
    Accepts dictionary of models to test, using default parameters unless otherwise specified.
    Models and parameters dictionaries must have matching keys
    Currently assumes X has been scaled, normalized, or transformed as needed.
    '''
    results = defaultdict(str)
    
    for name, model in models.items():
        cv_score = np.mean(cross_val_score(model(**params[name]),X,y,cv=use_cv, scoring = metric))
        
        if verbose:
            print('Model:', name, 'Metric:', metric, cv_score)
                  
        results[name] = cv_score
    
    return results

def cross_val_xgb(X,y, folds, cv_scorer, pred_threshold=0.5, fit_metric='auc',
                  model_objective='binary:logistic'):
    '''
    Performs cross-validation on an XGBoost estimator object. Fits a model on
    each fold of the provided data.
    Returns cross-validation error measurements.
    ---
    Inputs:
        X: pandas dataframe, columns are features and rows are training examples
        y: pandas series, labels for examples in X.
        folds: generator that produces indices for cross-validation
        cv_scorer: error metric function
        pred_threshold: value at which to predict 1 or 0
        fit_metric, model_objective: XGBoost parameters
    Returns:
        Mean score of cv_scorer across all folds.
    '''
    
    def prob_to_pred(num, cutoff=pred_threshold):
        # Converts xgb prediction output to binary value
        return 1 if num > cutoff else 0
    
    # Prepare to store individual fold scores
    cv_scores = []
    
    # Fit model for each fold and retain error metrics
    for train_idx, val_idx in folds.split(X, y):
        
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
       
        gbm = xgb.XGBRegressor( 
                           n_estimators=30000, #arbitrary large number b/c we're using early stopping
                           max_depth=3,
                           objective= model_objective,
                           learning_rate=.1, 
                           subsample=1,
                           min_child_weight=1,
                           colsample_bytree=.8
                          )
        eval_set=[(X_tr,y_tr),(X_val,y_val)] #tracking train/validation error as we go
        fit_model = gbm.fit( 
                        X_tr, y_tr, 
                        eval_set=eval_set,
                        eval_metric=fit_metric,
                        early_stopping_rounds=50, # stop when validation error hasn't improved in this many rounds
                        verbose=False #gives output log as below
                       )
        # Make and assess validation predicitons
        y_pred = pd.Series(fit_model.predict(X_val,
            ntree_limit=gbm.best_ntree_limit)).apply(prob_to_pred)
        
        cv_scores.append(cv_scorer(y_val,y_pred))
    
    # Calculate CV Error and mean custom error if applicable
    return np.mean(cv_scores)

def fit_score_model(features,target,model=DummyClassifier, metric=accuracy_score, params = {}):
    '''
    Function for creating a model instance and getting the desired classification error metric.
    Defaults to sklearn dummy classifier with default parameters
    ----
    Inputs:
    features: array-like object containing features for model fitting
    target: 1-D array-like object cotaining labels for model fitting
    model: sklearn model class or similar that estimates the target value from training data
    metric: function for evaluating an estimator against the true labels.
            Should take arguments (y_true, y_pred) in that order.
    
    Returns:
    score: float value output by metric
    '''
    # Initiate model instance with relevant parameters
    instance = model(**params)
    
    instance.fit(features,target)
    
    # Determine whether to use labels or probabilities when evaluating model
    proba_based_metrics = [log_loss]
    
    if metric in proba_based_metrics:
        return metric(target, instance.predict_proba(features))
    else:
        return metric(target, instance.predict(features))


def iterate_k_for_KNN(X, y, start_k, stop_k, metric='roc_auc',use_cv=5):
    '''
    Tests k-nearest neighbors algorithm for provided range of k.
    Returns k value that optimizes for the provided classification error metric, and the error score.
    '''
    best_k, best_score = None, 0
    for k in range(start_k,stop_k):
        knn_params = {'n_neighbors':k}
        score = np.mean(cross_val_score(KNeighborsClassifier(**knn_params),X,y,
                            cv=use_cv, scoring = metric))    
        print('n_neighbors:',k,metric,score)

        if score > best_score:
            best_k = k
            best_score = score

    return(best_k,best_score)
        