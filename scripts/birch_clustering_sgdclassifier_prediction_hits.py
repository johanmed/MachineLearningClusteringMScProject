#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies:
- vector_data.py -> data, preprocessing_hits
- general_clustering -> ModellingBirch
Birch is run on training data to get clustering
SGDClassifier is run on clusters extracted by Birch to predict clustering and description or trait category of validation data
Modelling by hits (chromosome number + marker position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_hits

import numpy as np

y_train=X_train['desc']

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_valid=X_valid['desc']

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_test=X_test['desc']

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]



# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import Birch # import Birch class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.linear_model import SGDClassifier # import SGDClassifer for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from general_clustering import ModellingBirch


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering(ModellingBirch):

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
        


    def perform_birch(self):
        """
        Perform Birch clustering on 2 features columns
        """
        birch_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('birch', Birch())])
        birch_clustering.fit(self.training) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', birch_clustering.labels_[:5]) # check labels of first 5 training data
        
        y_pred=birch_clustering.predict(self.validation)
        
        return birch_clustering, y_pred
    
    
    
    def visualize_plot(plot_birch, birch_clustering, X_train, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_birch(birch_clustering, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_BIRCH_clustering_SVM_prediction_result_by_hits"))
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
        plt.savefig(os.path.join(out_dir, f"Project_PCA_BIRCH_clustering_SGD__{type_anno}annotation_result_by_hits"))
        plt.show()

     
     
# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

    actual_clustering, prediction_clusters_valid=clustering_task.perform_birch()

    Columns2Clustering.visualize_plot(Columns2Clustering.plot_birch, actual_clustering[1], X_train_features)

    Columns2Clustering.visualize_plot(Columns2Clustering.plot_birch, actual_clustering[1], X_valid_features)

    extracted_annotation=Columns2Clustering.extract_features_target_relationship(X_train_features, y_train, X_valid_features, y_valid)

    Columns2Clustering.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Columns2Clustering.visualize_plot_annotation(X_valid_features, y_valid, 'actual')


import timeit

time_taken = timeit.timeit(lambda: main(), number=10)
print(f"Execution time for dbscan_clustering_svm_prediction_hits.py is : {time_taken} seconds")


