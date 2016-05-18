# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:19:07 2016

@author: Bishwarup
"""
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, rankdata

parent_dir = "F:/Kaggle/Santander/"
output_dir = os.path.join(parent_dir, "Submissions/")

glmnet2 = pd.read_csv(os.path.join(parent_dir, 'Branden/L2_testPreds_glmnet2.csv'))
xgb3 = pd.read_csv(os.path.join(output_dir, 'stacked_xgb_3.csv')) 

m = pd.merge(glmnet2, xgb3, on = 'ID')
print spearmanr(m.TARGET_x, m.TARGET_y)
m['rank_glm'] = rankdata(m.TARGET_x)
m['rank_xgb'] = rankdata(m.TARGET_y)
m['avg_rank'] = 0.55*m.rank_glm + 0.45*m.rank_xgb

scalar = MinMaxScaler()
m['TARGET'] =scalar.fit_transform(m.avg_rank)
sub = m[['ID', 'TARGET']]
sub.to_csv(os.path.join(output_dir, 'L2_glm_xgb_rank_average.csv'), index = False)


glmnet_xgb = pd.read_csv(os.path.join(output_dir, 'L2_glm_xgb_rank_average.csv')).rename(columns = {'TARGET' : 'glmnet_xgb'})
bb = pd.read_csv(os.path.join(parent_dir, 'Mohamed/predictions_best_1.csv')).rename(columns = {'TARGET': 'bb'})
m = pd.merge(glmnet_xgb, bb, on = 'ID')
print spearmanr(m.glmnet_xgb, m.bb)
m['rank_glmnet_xgb'] = rankdata(m.glmnet_xgb)
m['rank_bb'] = rankdata(m.bb)
m['avg_rank'] = 0.61*m.rank_glmnet_xgb + 0.45*m.rank_bb
scalar = MinMaxScaler()
m['TARGET'] =scalar.fit_transform(m.avg_rank)
sub = m[['ID', 'TARGET']]
sub.to_csv(os.path.join(output_dir, 'wra_bsubs.csv'), index = False)


###
parent_dir = "F:/Kaggle/Santander/Branden"
glm8 = pd.read_csv(os.path.join(parent_dir, 'L2_testPreds_glmnet8.csv')).rename(columns = {'TARGET': 'glm8'})
xgb3 = pd.read_csv(os.path.join(parent_dir, 'stacked_xgb_3.csv')).rename(columns = {'TARGET': 'xgb3'})
nn = pd.read_csv(os.path.join(parent_dir, 'L2_nn.csv')).rename(columns = {'TARGET' : 'nn'})

m = pd.merge(glm8, xgb3, on = 'ID')
m = pd.merge(m, nn, on = 'ID')
print spearmanr(m.glm8, m.xgb3)
print spearmanr(m.xgb3, m.nn)
m['rank_glm8'] = rankdata(m.glm8)
m['rank_xgb3']= rankdata(m.xgb3)
m['rank_nn'] = rankdata(m.nn)

m['r_xgb3_nn'] = 0.75*m['rank_xgb3'] + 0.25*m['rank_nn']
m['r_3'] = 0.59*m['rank_glm8'] + 0.41*m['r_xgb3_nn']
scalar = MinMaxScaler()
m['TARGET'] = scalar.fit_transform(m['r_3'])
sub = m[['ID', 'TARGET']]
sub.to_csv(os.path.join(output_dir, 'RA_l2_glmnet8_nn_xgb3.csv'), index = False)