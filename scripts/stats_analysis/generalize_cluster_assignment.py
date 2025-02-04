#!/usr/bin/env python
"""
Add a new column y to dataset for the cluster using the best classical model for clustering
"""

from vector_data_pre import preprocessing_hits, preprocessing_qtl

import pandas as pd

from sklearn.pipeline import Pipeline

import joblib

import os

# 1. Read in original data

full_X_rec=pd.read_csv('../../../../project_dataset_all_traits_p_lrt_filtered.csv', usecols=['chr_num', 'pos', 'p_lrt'])

#print('full_X_rec looks like: \n', full_X_rec)

def predict_cluster(full_X_rec, preprocessing_type, model):
    """
    Fit the same pipeline to the complete dataset
    Predict the cluster of each observation
    """
    clustering_model=Pipeline([('preprocessing_type', preprocessing_type), ('model', model)])
    
    clustering_model.fit(full_X_rec)
    
    clusters_predicted=clustering_model.predict(full_X_rec)
    
    return clusters_predicted


# 2. Use birch hits and qtl models to extract clusters

if os.path.exists('../clustering/birch_clustering/birch_clustering_hits.pkl'):

    model_birch=joblib.load('../clustering/birch_clustering/birch_clustering_hits.pkl')
    
    y_birch_hits=predict_cluster(full_X_rec, preprocessing_hits, model_birch)
    
    full_X_rec['y_hits']=y_birch_hits


if os.path.exists('../clustering/birch_clustering/birch_clustering_qtl.pkl'):
    
    model_birch=joblib.load('../clustering/birch_clustering/birch_clustering_qtl.pkl')
    
    y_birch_qtl=predict_cluster(full_X_rec, preprocessing_hits, model_birch)
    
    full_X_rec['y_qtl']=y_birch_qtl


#print('Extended full_X_rec looks like:\n', full_data)


