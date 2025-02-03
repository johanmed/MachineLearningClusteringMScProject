#!/usr/bin/env python

"""
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies:
- vector_data_pre.py -> data, preprocessing_hits
- general_clustering -> ModellingKMeans
KMeans is run twice:
1. Identify the best number of clusters for the data using the training data
2. Proceed to actual training and validation on respective data
Modelling by hits (chromosome number + chromosomal position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

import os

from vector_data_pre import scaled_training_set as X_train
from vector_data_pre import scaled_validation_set as X_valid
from vector_data_pre import scaled_test_set as X_test

from vector_data import preprocessing_hits

import pandas as pd
import numpy as np

X_train=X_train[['p_lrt', 'chr_num', 'pos']]

X_valid=X_valid[['p_lrt', 'chr_num', 'pos']]

X_test=X_test[['p_lrt', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import MiniBatchKMeans # import MiniBatchKMeans class
import matplotlib.pyplot as plt # import plot manager
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from general_clustering import ModellingKMeans


out_dir=os.path.abspath('../../output/') # define directory to save plots to




class Columns2Clustering(ModellingKMeans):

    """
    Represent clustering task on only 2 features extracted from dimensionality reduction
    """
    
    def get_features(self):
        """
        Extract 2 PCA from preprocessing_hits pipeline
        """
        preprocessed_training=preprocessing_hits.fit_transform(self.training)
        preprocessed_validation=preprocessing_hits.transform(self.validation)
        preprocessed_test=preprocessing_hits.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_kmeans_clustering(self, reduced_features_valid):
        """
        Run KMeans for number of clusters on training and save predictions and distances to centroids
        """
        kmeans_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('kmeans', MiniBatchKMeans(random_state=2024))])
        kmeans_clustering.fit(self.training)
        #print('The labels assigned to the following training data \n', X_train[:5], ' are respectively: \n', kmeans.labels_[:5]) # check labels of first 5 training data
        y_pred=kmeans_clustering.predict(self.validation)
        #print('The labels assigned to the following validation data \n', X_valid, ' are respectively: \n', y_pred[:5]) # check labels assigned to first 5 validation data

        distance_inst_centro=kmeans_clustering.transform(self.training).round(2) # Save distance of instances to centroids infered for the best number of clusters
        #print('The distances to each centroid for the first 5 instances are: \n', distance_inst_centro[:5]) # can change to see for more
        
        print('The silhouette score obtained as clustering performance measure is:', silhouette_score(reduced_features_valid, y_pred))
        
        return kmeans_clustering, y_pred, distance_inst_centro



    def plot_kmeans(clusterer, X, plot_decision_boundaries):
        """
        Plot clusters extracted by KMeans
        """
                
        plot_decision_boundaries(clusterer, X)
        

    def visualize_plot(plot_kmeans, clusterer, X_train):
        """
        Generate actual visualization of clusters
        Save figure
        """
        
        plt.figure(figsize=(10, 10))
        plot_kmeans(clusterer, X_train, Columns2Clustering.plot_decision_boundaries)
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_clustering_result_by_hits"), dpi=500)
        
        
  

# Main

import joblib

def main():
    
    if os.path.exists('kmeans_clustering/kmeans_clustering_hits.pkl'):
        
        print('The model has already been trained and saved on disk!')
        
    else:
    
        clustering_task=Columns2Clustering(X_train, X_valid, X_test)

        X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

        actual_clustering=clustering_task.perform_kmeans_clustering(X_valid_features)
        
        joblib.dump(actual_clustering[0][1], 'kmeans_clustering/kmeans_clustering_hits.pkl')

        #Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[0][1], X_train_features)

        Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[0][1], X_valid_features)

        prediction_clusters=actual_clustering[1]

        distances_centroids_validation=actual_clustering[2]



main()
