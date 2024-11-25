#!/usr/bin/env python

"""
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies:
- vector_data.py -> data, preprocessing_qtl
- general_clustering -> ModellingDBSCAN
KMeans is run twice:
1. Identify the best number of clusters for the data using the training data
2. Proceed to actual training and validation on respective data
SGDClassifier is run to predict description or trait category of validation data
Modelling by QTL (chromosome number)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_qtl

import numpy as np

y_train=X_train['desc']

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]

y_valid=X_valid['desc']

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]

y_test=X_test['desc']

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]



# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import KMeans # import KMeans class
from sklearn.metrics import silhouette_score # import silhouette_score class
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.linear_model import SGDClassifier # import SGDClassifer for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from general_clustering import ModellingKMeans


out_dir=os.path.abspath('../output/') # define directory to save plots to




class Columns2Clustering(ModellingKMeans):

    """
    Represent clustering task on only 2 features extracted from dimensionality reduction
    """
    
    def get_features(self):
        """
        Extract 2 PCA from preprocessing_qtl pipeline
        """
        preprocessed_training=preprocessing_qtl.fit_transform(self.training)
        preprocessed_validation=preprocessing_qtl.transform(self.validation)
        preprocessed_test=preprocessing_qtl.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_kmeans_clustering(self):
        """
        Run KMeans for number of clusters on training and save predictions and distances to centroids
        """
        kmeans_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('kmeans', KMeans(algorithm='elkan', random_state=2024))])
        kmeans_clustering.fit(self.training)
        #print('The labels assigned to the following training data \n', X_train[:5], ' are respectively: \n', kmeans.labels_[:5]) # check labels of first 5 training data
        y_pred=kmeans_clustering.predict(self.validation)
        #print('The labels assigned to the following validation data \n', X_valid, ' are respectively: \n', y_pred[:5]) # check labels assigned to first 5 validation data

        distance_inst_centro=kmeans_clustering.transform(self.training).round(2) # Save distance of instances to centroids infered for the best number of clusters
        #print('The distances to each centroid for the first 5 instances are: \n', distance_inst_centro[:5]) # can change to see for more
        
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
        plt.savefig(os.path.join(out_dir, f"Project_PCA_KMeans_clustering_SVM_prediction_result_by_qtl"))
        plt.show()
        
        
    def extract_features_target_relationship(X_train, y_train, X_valid, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target)
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        assign_vec=SGDClassifier(random_state=2024)
        assign_vec.fit(X_train, y_train)
        y_supervised_pred=assign_vec.predict(X_valid)
        #print('The prediction for the first 5 validation data is :', y_pred[:5])
        print(classification_report(y_valid, y_supervised_pred))
        
        return y_supervised_pred


    def visualize_plot_annotation(X_valid, y_supervised_pred, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_supervised_pred)
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_PCA_KMeans_clustering_SGD_{type_anno}_annotation_result_by_qtl"))
        plt.show()



# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

    actual_clustering=clustering_task.perform_kmeans_clustering()

    Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[0][1], X_train_features)
    
    Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[0][1], X_valid_features)

    prediction_clusters=actual_clustering[1]

    distances_centroids_validation=actual_clustering[2]

    extracted_annotation=Columns2Clustering.extract_features_target_relationship(X_train_features, y_train, X_valid_features, y_valid)

    Columns2Clustering.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Columns2Clustering.visualize_plot_annotation(X_valid_features, y_valid, 'actual')





import timeit

time_taken = timeit.timeit(lambda: main(), number=10)
print(f"Execution time for kmeans_clustering_svm_prediction_qtl.py is : {time_taken} seconds")
