# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:05:20 2016

@author: Bishwarup
"""

from __future__ import division
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier

parent_dir = 'F:/Kaggle/Santander'
eval_dir = os.path.join(parent_dir, 'evals/')
test_dir = os.path.join(parent_dir, 'tests/')
np.random.seed(112)
pred_name = 'ada1'

if __name__ == '__main__': 
    
    start_time = datetime.now()
    # read data
    alldata = pd.read_csv(os.path.join(parent_dir, 'glmnetVarSelectAlldata.csv'))
    fold_ids = pd.read_csv(os.path.join(parent_dir, 'Fold5F.csv'))
    #
    #
    train = alldata[alldata['filter'] == 0].copy()
    test = alldata[alldata['filter'] == 2].copy().reset_index(drop = True)
    #
    feature_names = [f for f in alldata.columns if f not in ['ID', 'TARGET', 'filter']]
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
        
        ada = AdaBoostClassifier(n_estimators = 2000,
                                    learning_rate = 0.04,
                                    random_state = 121)        
        
        ada.fit(trainingSet[feature_names], np.array(trainingSet['TARGET']))
        preds = ada.predict_proba(validationSet[feature_names])[:, 1]
        auc = roc_auc_score(np.array(validationSet['TARGET']), preds)
        print 'oof auc: {}'.format(auc)
        
        tmp = pd.DataFrame({'ID': validationSet['ID'],  pred_name: preds})
        eval_matrix = eval_matrix.append(tmp, ignore_index = True)
        
        del trainingSet, validationSet, ada, auc, val_ids, idx
        gc.collect()         
    
    ada = AdaBoostClassifier(n_estimators = 2000,
                                    learning_rate = 0.04,
                                    random_state = 121)
                            
    ada.fit(train[feature_names], np.array(train['TARGET']))
    tpreds = ada.predict_proba(test[feature_names])[:, 1]
    test_matrix[pred_name] = tpreds
    
    # save to disk
    eval_matrix.to_csv(os.path.join(eval_dir, pred_name+'_eval.csv'), index = False)
    test_matrix.to_csv(os.path.join(test_dir, pred_name+'_test.csv'), index = False)
        
    end_time = datetime.now()
    print 'elapsed time: {}'.format(end_time - start_time)    