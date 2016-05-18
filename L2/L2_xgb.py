# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:53:30 2016

@author: Bishwarup
"""
import os
import gc
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from datetime import  datetime
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
parent_dir = "F:/Kaggle/Santander/"

eval_dir = os.path.join(parent_dir, "all_evals/")
test_dir = os.path.join(parent_dir, "all_tests/")
output_dir = os.path.join(parent_dir, "Submissions/")
np.random.seed(11)

def merge_all(id_, path, key = "ID"):
    merged_data = pd.DataFrame({key : id_})
    file_list = os.listdir(path)
    for files_ in file_list:
        candidate = pd.read_csv(os.path.join(path, files_))
        if "Fold" in candidate.columns:
            candidate.drop("Fold", axis = 1, inplace = True)
        if "ground_truth" in candidate.columns:
            candidate.drop("ground_truth", axis = 1, inplace = True)
        if "TARGET" in candidate.columns:
            print files_
            candidate.drop("TARGET", axis = 1, inplace = True)
        assert (len(id_) == candidate.shape[0]), "{0} have differnt number of rows!".format(files_)
        merged_data = pd.merge(merged_data, candidate, on = key, how = "left")
        
    print "merged {0} files ..".format(len(file_list))        
    return merged_data
    
def logit(x):
    
    lgt = 1/(1+np.exp(-x))
    return lgt
    
if __name__ == "__main__":
    
    print "reading L0 ..."
    train_file = os.path.join(parent_dir, "train_mrmr_xgbImp.csv")
    test_file = os.path.join(parent_dir, "test_mrmr_xgbImp.csv")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    xgb_imp = pd.read_csv(os.path.join(parent_dir, 'data_V1/feature_importance.csv'))
    print "merging L1..."
    train_preds = merge_all(id_=np.array(train["ID"]),path=eval_dir)
    test_preds = merge_all(id_=np.array(test["ID"]),path=test_dir)

    train_preds.ix[:, 1:] = np.log(train_preds.ix[:, 1:]/(1-train_preds.ix[:, 1:]))
    test_preds.ix[:, 1:] = np.log(test_preds.ix[:, 1:]/(1-test_preds.ix[:, 1:]))
    #assert (), 'the meta feature names does not match!'
    print "merging L0 and L1..."
    train = pd.merge(train, train_preds, on = "ID", how = "left")
    test = pd.merge(test, test_preds, on = "ID", how = "left")
           
    feature_names = list(xgb_imp.Feature)[:20]
    meta = [f for f in train_preds.columns if f not in['ID']]
    feature_names = feature_names + meta
    
    feature_names =  meta
    feature_names.remove('h2o_gbm_L1_v1')
    feature_names.remove('h2o_rf_L1_v1')
    #feature_names = [f for f in train.columns if f not in ["ID", "target", "train_flag"]]
    
    skf = StratifiedKFold(np.array(train["TARGET"]), n_folds = 10, shuffle = True, random_state = 14) 

    cv = []
    biter = []    
    for fold, (itr, icv) in enumerate(skf):
    
        print "------ Fold %d -----------\n" %(fold+1)
        
        trainingSet = train.iloc[itr]
        validationSet = train.iloc[icv]
        
        gbm = XGBClassifier(max_depth=4,
                            learning_rate = 0.01,
                            n_estimators=3000,
                            subsample=0.8,
                            colsample_bytree=0.5,
                            objective="binary:logistic",
                            silent = False,
                            min_child_weight=5,                       
                            nthread=-1)
                            
        gbm.fit(trainingSet[feature_names], np.array(trainingSet["TARGET"]),
                eval_metric="auc",
                eval_set=[(trainingSet[feature_names], np.array(trainingSet["TARGET"])), (validationSet[feature_names], np.array(validationSet["TARGET"]))],
                         early_stopping_rounds=200,verbose=20)    
                          
        ll = gbm.best_score
        best_iter = gbm.best_iteration
        cv.append(ll)
        biter.append(best_iter)
        print "---auc : %0.6f\n" %ll
        print "---best_iter: %d\n" %best_iter
        gc.collect()
    
    gbm = XGBClassifier(max_depth=4,
                            learning_rate = 0.01,
                            n_estimators=370,
                            subsample=0.8,
                            colsample_bytree=0.5,
                            objective="binary:logistic",
                            silent = False,
                            min_child_weight=5,                       
                            nthread=-1)
                            
    gbm.fit(train[feature_names], np.array(train["TARGET"]),
            eval_metric = "auc",
            eval_set = [(train[feature_names], np.array(train["TARGET"]))],
                        verbose=20)                            
                        
    tpreds = gbm.predict_proba(test[feature_names])[:, 1]
    df = pd.DataFrame({"ID" : test["ID"], "TARGET" : tpreds })
    submission_name = "stacked_xgb_3.csv"
    df.to_csv(os.path.join(output_dir, submission_name), index = False)
