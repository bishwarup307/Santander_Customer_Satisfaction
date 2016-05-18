# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:11:30 2016

@author: Bishwarup
"""
from __future__ import division
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

parent_dir = 'F:/Kaggle/Santander'
eval_dir = os.path.join(parent_dir, 'evals/')
test_dir = os.path.join(parent_dir, 'tests/')
np.random.seed(112)
pred_name = 'xgb_c1'

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
        
        bst = XGBClassifier(objective = 'binary:logistic',
                            max_depth = 4,
                            learning_rate= 0.01,
                            subsample=0.8,
                            colsample_bytree=0.4,
                            n_estimators=1650,
                            min_child_weight=1,
                            silent=False)
        
        bst.fit(trainingSet[feature_names], np.array(trainingSet['TARGET']),
                eval_metric='auc',
                eval_set=[(trainingSet[feature_names], trainingSet['TARGET']), (validationSet[feature_names], validationSet['TARGET'])],
                          verbose=100)
        
        preds = bst.predict_proba(validationSet[feature_names])[:, 1]
        tmp = pd.DataFrame({'ID': validationSet['ID'],  pred_name: preds})
        eval_matrix = eval_matrix.append(tmp, ignore_index = True)
        
        del trainingSet, validationSet, bst, val_ids, idx
        gc.collect()         
    
    bst = XGBClassifier(objective = 'binary:logistic',
                            max_depth = 4,
                            learning_rate= 0.01,
                            subsample=0.8,
                            colsample_bytree=0.4,
                            n_estimators=1650,
                            min_child_weight=1,
                            silent=False,
                            nthread=-1)       
                            
    bst.fit(train[feature_names], np.array(train['TARGET']),
            eval_metric='auc',
            eval_set=[(train[feature_names], train['TARGET'])],
                      verbose=100)
    tpreds = bst.predict_proba(test[feature_names])[:, 1]
    test_matrix[pred_name] = tpreds
    
    # save to disk
    eval_matrix.to_csv(os.path.join(eval_dir, pred_name+'_eval.csv'), index = False)
    test_matrix.to_csv(os.path.join(test_dir, pred_name+'_test.csv'), index = False)
        
    end_time = datetime.now()
    print 'elapsed time: {}'.format(end_time - start_time)    