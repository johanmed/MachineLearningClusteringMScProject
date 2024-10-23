#!/usr/bin/env python
"""
Summary:
This script contains code to do clustering using neural networks (shallow/deep learning)
Dependencies: 
- vector_data.py -> data, preprocessing_hits
Neural networks are used to get clustering and to predict description or trait category of validation data
Modelling by hits (chromosome number + marker position)
"""


# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_hits

import numpy as np

y_train=np.array(X_train['desc'][:5000])

X_train=np.array(X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:5000]) # select 5000 first columns of combined and transformed columns

y_valid=np.array(X_valid['desc'][:5000])

X_valid=np.array(X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:5000]) # same

y_test=np.array(X_test['desc'][:5000])

X_test=np.array(X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:5000]) # same


# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf # import DBSCAN class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering:

    """
    Represent clustering task on 2 columns extracted from dimensionality reduction
    """
    
    def __init__(self, training, validation, test):
        self.training=training
        self.validation=validation
        self.test=test

    def perform_neural_clustering(self):
        """
        Perform neural clustering on 2 features columns
        """
        neural_model_unsup=tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu, input_dim=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.softmax),
        ])
        neural_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('neural_unsupervised', neural_model_unsup])
        neural_clustering.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        neural_clustering.fit(x=self.training, epochs=20)
        #print('The labels for the first 5 training data are: ', dbscan_clustering.labels_[:5]) # check labels of first 5 training data
        
        return neural_clustering
    
    
    
    def visualize_plot(neural_clustering, X_train, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        
        y_clustering_pred_training=neural_clustering.predict(X_train)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(neural_clustering[0][0], neural_clustering[0][1], c=y_clustering_pred_training)
        plt.xlabel("PCA 1", fontsize=10)
        plt.ylabel("PCA 2", fontsize=10, rotation=90)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_neural_clustering_result_by_hits"))
        plt.show()
        
        
     
    def predict_neural_clustering(neural_clustering, X_valid):
        """
        Use neural networks to predict clustering on validation set
        """
        y_clustering_pred_valid=neural_clustering.predict(X_valid)
        
        def extract_dist(features, labels):
           """
           Takes dbscan labels
           Calculates the distance of each data to cluster
           Returns numpy arrays of distances and corresponding indices
           """
           # Get centers
           clusters_centers=[]
           for label in set(labels):
               if label != -1:
                   cluster_center=np.mean(features[labels==label], axis=0)
                   clusters_centers.append(cluster_center)
        
           # Calculate all distances to each cluster center
           all_distances=np.zeros((len(features), len(clusters_centers)))
           for i, center in enumerate(clusters_centers):
               all_distances[:, i]=np.linalg.norm(features-center, axis=1)
       
           # select distance to the nearest neighbor only
           distance=[]
           for element in all_distances:
               distance.append(min(element))
       
           return np.array(distance)
            
        y_dist=extract_dist(X_valid, y_clustering_pred_valid) # get distances and indices to nearest clusters obtained for training data
        y_clustering_pred_valid[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
        #print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data
        return y_clustering_pred_valid


     
    def extract_features_target_relationship(X_train, y_train, X_valid, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target) using neural networks
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        neural_model_sup=tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu, input_dim=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=2, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=3, activation=tf.nn.softmax),
        ])
        neural_assign=Pipeline([('preprocessing_hits', preprocessing_hits), ('neural_supervised', neural_model_sup])
        neural_assign.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        neural_assign.fit(x=X_train, y=y_train, epochs=20)
        y_supervised_pred=neural_assign.predict(X_valid)
        #print('The prediction for the first 5 validation data is :', y_pred[:5])
        print(classification_report(y_valid, y_supervised_pred))
        
        return y_supervised_pred


    def visualize_plot_annotation(neural_clustering, X_valid, y_supervised_pred, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(neural_clustering[0][0], neural_clustering[0][1], c=y_supervised_pred)
        plt.xlabel("PCA 1", fontsize=10)
        plt.ylabel("PCA 2", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_PCA_DBSCAN_clustering_KNN_{type_anno}_annotation_result_by_hits"))
        plt.show()

     


# Main

clustering_task=Columns2Clustering(X_train, X_valid, X_test)
actual_clustering=clustering_task.perform_dbscan_clustering()
Columns2Clustering.visualize_plot(Columns2Clustering.plot_dbscan, actual_clustering, X_train)
prediction_clusters=Columns2Clustering.predict_dbscan_clustering(actual_clustering, X_valid)
extracted_annotation=Columns2Clustering.extract_features_target_relationship(X_train, y_train, X_valid, y_valid)
Columns2Clustering.visualize_plot_annotation(X_valid, extracted_annotation, 'predicted')
Columns2Clustering.visualize_plot_annotation(X_valid, y_valid, 'actual')



