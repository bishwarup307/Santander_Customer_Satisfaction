# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:29:23 2016

@author: Bishwarup
"""

from __future__ import division
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

parent_dir = 'F:/Kaggle/Santander'
eval_dir = os.path.join(parent_dir, 'evals/')
test_dir = os.path.join(parent_dir, 'tests/')
np.random.seed(112)
pred_name = 'sgd1'

if __name__ == '__main__': 
    # read data
    alldata = pd.read_csv(os.path.join(parent_dir, 'glmnetVarSelectAlldata.csv'))
    fold_ids = pd.read_csv(os.path.join(parent_dir, 'Fold5F.csv'))
    #
    use_cols = [f for f in alldata.columns if f not in ['ID', 'TARGET', 'filter']]
    use_df = alldata[use_cols].copy()
    scalar = StandardScaler()
    use_df = pd.DataFrame(scalar.fit_transform(use_df))
    use_df.columns = use_cols
    alldata.drop(use_cols, axis = 1, inplace = True)
    alldata = pd.concat([alldata, use_df], axis = 1)
    #
    train = alldata[alldata['filter'] == 0].copy()
    test = alldata[alldata['filter'] == 2].copy().reset_index(drop = True)
    
    #
    feature_names = use_cols
    #
    eval_matrix = pd.DataFrame(columns = ['ID', pred_name])
    test_matrix = pd.DataFrame({'ID' : test['ID']})
    #
    for ii in xrange(fold_ids.shape[1]):
        
        print '---- fold : {} ------'.format(ii + 1)
        val_ids = fold_ids.ix[:, ii].dropna()
        idx = train["ID"].isin(list(val_ids))
        trainingSet = train[~idx]
        validationSet = train[idx]
        
        sgd = SGDClassifier(loss = 'log',
                            penalty = 'l1',
                            alpha = 0.0002,
                            n_iter = 60,
                            fit_intercept = True,
                            shuffle = True,
                            random_state = 10,
                            n_jobs = -1,
                            verbose = 0)        
        
        sgd.fit(trainingSet[feature_names], np.array(trainingSet['TARGET']))
        preds = sgd.predict_proba(validationSet[feature_names])[:, 1]
        auc = roc_auc_score(np.array(validationSet['TARGET']), preds)
        print 'oof auc: {}'.format(auc)
        
        tmp = pd.DataFrame({'ID': validationSet['ID'],  pred_name: preds})
        eval_matrix = eval_matrix.append(tmp, ignore_index = True)
        
        del trainingSet, validationSet, sgd, auc, val_ids, idx
        gc.collect()         
    
    sgd = SGDClassifier(loss = 'log',
                            penalty = 'l1',
                            alpha = 0.0002,
                            n_iter = 60,
                            fit_intercept = True,
                            shuffle = True,
                            random_state = 10,
                            n_jobs = -1,
                            verbose = 1)
                            
    sgd.fit(train[feature_names], np.array(train['TARGET']))
    tpreds = sgd.predict_proba(test[feature_names])[:, 1]
    test_matrix[pred_name] = tpreds
    
    # save to disk
    eval_matrix.to_csv(os.path.join(eval_dir, pred_name+'_evals.csv'), index = False)
    test_matrix.to_csv(os.path.join(test_dir, pred_name+'_test.csv'), index = False)
        
    