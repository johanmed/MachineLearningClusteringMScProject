#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies: vector_data.py -> data
DBSCAN is run on training data to get clustering
KNeighborsClassifier is run on clusters extracted by DBSCAN to predict clustering and description or trait category of validation data
Modelling by hits (chromosome number + marker position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test

import numpy as np

y_train=X_train['desc'][:5000]

X_train=X_train[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # select 5000 first columns of combined and transformed columns

y_valid=X_valid['desc'][:5000]

X_valid=X_valid[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same

y_test=X_test['desc'][:5000]

X_test=X_test[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same




# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import DBSCAN # import DBSCAN class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.neighbors import KNeighborsClassifier # import KNeighborsClassifier for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering:

    """
    Represent clustering task on 2 specific features columns
    """
    
    def __init__(self, training, validation, test, index1, index2):
        """
        Instantiate a class object
        """
        self.training=training
        self.validation=validation
        self.test=test
        self.index1=index1
        self.index2=index2
        
        
    def get_all_datasets(self):
        """
        Return new datasets with only the specific features columns selected
        """
        return np.array(self.training[[f'transformed_combined_chr_num_pos{self.index1}', f'transformed_combined_desc_p_lrt{self.index2}']]), np.array(self.validation[[f'transformed_combined_chr_num_pos{self.index1}', f'transformed_combined_desc_p_lrt{self.index2}']]), np.array(self.test[[f'transformed_combined_chr_num_pos{self.index1}', f'transformed_combined_desc_p_lrt{self.index2}']])


    def perform_dbscan_clustering(self):
        """
        Perform DBSCAN clustering on 2 features columns
        """
        dbscan=DBSCAN(eps=0.25)
        dbscan.fit(np.array(self.training[[f'transformed_combined_chr_num_pos{self.index1}', f'transformed_combined_desc_p_lrt{self.index2}']])) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', dbscan.labels_[:5]) # check labels of first 5 training data
        return dbscan
    
    
    def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
        """
        Display DBSCAN clustering distinguishing, core, non core and anomalies instances
        Data plotted according to 2 features provided
        """
        core_mask=np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_]= True
        anomalies_mask=dbscan.labels_== -1
        non_core_mask= ~(core_mask | anomalies_mask)
        cores=dbscan.components_
        anomalies=X[anomalies_mask]
        non_cores=X[non_core_mask]
    
        plt.scatter(cores[:, 0], cores[:, 1], c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    
        if show_xlabels:
            plt.xlabel("Transformed trait category and p-lrt", fontsize=10)
        else:
            plt.tick_params(labelbottom=False)
    
        if show_ylabels:
            plt.ylabel("Transformed chromosome number and position", fontsize=10, rotation=90)
        else:
            plt.tick_params(labelleft=False)
    
    
    
    def visualize_plot(plot_dbscan, dbscan, X_train, index1, index2, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_dbscan(dbscan, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_DBSCAN_clustering_KNN_prediction_result_by_hits_{index1}_{index2}"))
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


    def visualize_plot_annotation(X_valid, y_supervised_pred, index1, index2, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_supervised_pred)
        plt.xlabel("Transformed trait category and p-lrt", fontsize=10)
        plt.ylabel("Transformed chromosome number and position", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_DBSCAN_clustering_KNN_{type_anno}_annotation_result_by_hits_{index1}_{index2}"))
        plt.show()

     
     
def columns2clustering(index1, index2):
    """
    Perform all clustering operations predefined for the indexes at hand
    """
    clustering_task=Columns2Clustering(X_train, X_valid, X_test, index1, index2)
    datasets=clustering_task.get_all_datasets()
    actual_clustering=clustering_task.perform_dbscan_clustering()
    Columns2Clustering.visualize_plot(Columns2Clustering.plot_dbscan, actual_clustering, datasets[0], index1, index2)
    prediction_clusters=Columns2Clustering.predict_dbscan_clustering(actual_clustering, datasets[1])
    extracted_annotation=Columns2Clustering.extract_features_target_relationship(datasets[0], y_train, datasets[1], y_valid)
    Columns2Clustering.visualize_plot_annotation(datasets[1], extracted_annotation, index1, index2, 'predicted')
    Columns2Clustering.visualize_plot_annotation(datasets[1], y_valid, index1, index2, 'actual')

for i in range(1, 6):
    for j in range(1, 6):
        columns2clustering(i, j)
        



