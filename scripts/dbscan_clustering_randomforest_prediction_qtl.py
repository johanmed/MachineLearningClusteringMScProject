#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies:
- vector_data.py -> data, preprocessing_qtl
- general_clustering -> ModellingDBSCAN
DBSCAN is run on training data to get clustering
RandomForestClassifier is run on clusters extracted by DBSCAN to predict clustering and description or trait category of validation data
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
from sklearn.ensemble import RandomForestClassifier # import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from general_clustering import ModellingDBSCAN


out_dir=os.path.abspath('../output/') # define directory where plots will be saved


class Columns2Clustering(ModellingDBSCAN):

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
        plt.savefig(os.path.join(out_dir, f"Project_PCA_DBSCAN_clustering_RandomForest_prediction_result_by_qtl"))
        plt.show()
        
        
     
    def predict_dbscan_clustering(dbscan, X_valid):
        """
        Run RandomForestClassifier on DBSCAN components and labels for prediction
        """
        pred_rand_for=RandomForestClassifier(random_state=2024)
        pred_rand_for.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train RandomForestClassifier on data and labels extracted from DBSCAN

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
    
        y_dist=extract_dist(X_valid, dbscan.labels_) # get distances and indices to nearest clusters obtained for training data
        y_clustering_pred=pred_rand_for.predict(X_valid) # get labels for validation data based on nearest cluster
        y_clustering_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
        #print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data
        
        return y_clustering_pred

     
    def extract_features_target_relationship(X_train, y_train, X_valid, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target)
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        assign_rand_for=RandomForestClassifier(random_state=2024)
        assign_rand_for.fit(X_train, y_train)
        y_supervised_pred=assign_rand_for.predict(X_valid)
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
        plt.savefig(os.path.join(out_dir, f"Project_PCA_DBSCAN_clustering_RandomForest_{type_anno}_annotation_result_by_qtl"))
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
print(f"Execution time for dbscan_clustering_randomforest_prediction_qtl.py is : {time_taken} seconds")


