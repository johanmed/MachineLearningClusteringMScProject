#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies:
- vector_data.py -> data, preprocessing_qtl
- general_clustering -> ModellingDBSCAN
DBSCAN is run on training data to get clustering
KNeighborsClassifier is run on clusters extracted by DBSCAN to predict clustering and description or trait category of validation data
Modelling by QTL peaks (chromosome number)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_qtl

import numpy as np

y_train=X_train['desc'][:5000]

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:5000] # select 5000 first columns of combined and transformed columns

y_valid=X_valid['desc'][:5000]

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:5000] # same

y_test=X_test['desc'][:5000]

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:5000] # same




# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import DBSCAN # import DBSCAN class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.neighbors import KNeighborsClassifier # import KNeighborsClassifier for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from general_clustering import ModellingDBSCAN


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering(ModellingDBSCAN):

    """
    Represent clustering task on only 2 columns extracted from dimensionality reduction
    """
    
    def get_features(self):
        """
        Extract 2 PCA from preprocessing_qtl pipeline
        """
        preprocessed_training=preprocessing_qtl.fit_transform(self.training)
        preprocessed_validation=preprocessing_qtl.transform(self.validation)
        preprocessed_test=preprocessing_qtl.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_dbscan_clustering(self):
        """
        Perform DBSCAN clustering on 2 features columns
        """
        dbscan_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('dbscan', DBSCAN(eps=0.25))])
        dbscan_clustering.fit(self.training) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', dbscan_clustering.labels_[:5]) # check labels of first 5 training data
        
        return dbscan_clustering
    
    
    
    def visualize_plot(plot_dbscan, dbscan, X_train, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_dbscan(dbscan, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_DBSCAN_clustering_KNN_prediction_result_by_qtl"))
        plt.show()
        
        
     
    def predict_dbscan_clustering(dbscan, X_valid):
        """
        Run KNeighborsClassfiers on DBSCAN components and labels for prediction
        """
        pred_knn=KNeighborsClassifier()
        
        pred_knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train knn on data and labels extracted from DBSCAN
        
        y_dist, y_pred_id=pred_knn.kneighbors(X_valid) # get distances and indices to nearest clusters obtained for validation data
        y_clustering_pred=dbscan.labels_[dbscan.core_sample_indices_][y_pred_id] # get labels for validation data based on nearest cluster
        y_clustering_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
        #print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data
        
        return y_clustering_pred



     
    def extract_features_target_relationship(X_train, y_train, X_valid, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target)
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        assign_knn=KNeighborsClassifier()
        assign_knn.fit(X_train, y_train)
        y_supervised_pred=assign_knn.predict(X_valid)
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
        plt.xlabel("PCA 1", fontsize=10)
        plt.ylabel("PCA 2", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_PCA_DBSCAN_clustering_KNN_{type_anno}_annotation_result_by_qtl"))
        plt.show()
     

# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

    actual_clustering=clustering_task.perform_dbscan_clustering()

    Columns2Clustering.visualize_plot(Columns2Clustering.plot_dbscan, actual_clustering[1], X_train_features)

    Columns2Clustering.visualize_plot(Columns2Clustering.plot_dbscan, actual_clustering[1], X_valid_features)

    prediction_clusters=Columns2Clustering.predict_dbscan_clustering(actual_clustering[1], X_valid_features)

    extracted_annotation=Columns2Clustering.extract_features_target_relationship(X_train_features, y_train, X_valid_features, y_valid)

    Columns2Clustering.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Columns2Clustering.visualize_plot_annotation(X_valid_features, y_valid, 'actual')




import timeit

time_taken = timeit.timeit(lambda: main(), number=10)
print(f"Execution time for dbscan_clustering_knn_prediction_qtl.py is : {time_taken} seconds")


