#!/usr/bin/env python

"""
Dependencies:
- vector_data.py -> data, preprocessing_qtl
- general_clustering -> ModellingBirch
Birch is run on training data to get clustering
Modelling by QTL peaks (chromosome number)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_qtl
import pandas as pd
import numpy as np


y_train=X_train['desc'][:(y_train.shape//2)]

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:(y_train.shape//2)]

y_valid=X_valid['desc'][:(y_valid.shape//2)]

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:(y_valid.shape//2)]

y_test=X_test['desc'][:(y_test.shape//2)]

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']][:(y_test.shape//2)]

X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import Birch # import Birch class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from general_clustering import ModellingBirch


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering(ModellingBirch):

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
        

    def perform_birch(self, reduced_features_valid):
        """
        Perform Birch clustering on 2 features columns
        """
        birch_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('birch', Birch())])
        birch_clustering.fit(self.training) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', birch_clustering.labels_[:5]) # check labels of first 5 training data
        
        y_pred=birch_clustering.predict(self.validation)
        
        print('The silhouette score obtained as clustering performance measure is:', silhouette_score(reduced_features_valid, y_pred))
        
        return birch_clustering, y_pred
    
    
    
    def visualize_plot(plot_birch, birch_clustering, X_train, size=200):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_birch(birch_clustering, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_BIRCH_clustering_result_by_qtl"))




# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    actual_clustering, prediction_clusters_valid=clustering_task.perform_birch(X_valid_features)

    #Columns2Clustering.visualize_plot(Columns2Clustering.plot_birch, actual_clustering[1], X_train_features)

    #Columns2Clustering.visualize_plot(Columns2Clustering.plot_birch, actual_clustering[1], X_valid_features)



import timeit

time_taken = timeit.timeit(lambda: main(), number=2)
print(f"Execution time for birch_clustering_qtl.py is : {time_taken} seconds")


