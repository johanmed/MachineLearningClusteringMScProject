#!/usr/bin/env python

"""
Dependencies: 
- vector_data.py -> data, preprocessing_hits
- general_clustering -> ModellingGaussian
BayesianGaussianMixture is run on training data to get clustering
Modelling by hits (chromosome number + marker position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_hits
import pandas as pd
import numpy as np

y_train=X_train['desc'][:(y_train.shape//2)]

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:(y_train.shape//2)]

y_valid=X_valid['desc'][:(y_valid.shape//2)]

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:(y_valid.shape//2)]

y_test=X_test['desc'][:(y_test.shape//2)]

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']][:(y_test.shape//2)]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.mixture import BayesianGaussianMixture # import BayesianGaussianMixture class for clustering
import matplotlib.pyplot as plt # import plot manager
from matplotlib.colors import LogNorm
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from general_clustering import ModellingGaussian

out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering(ModellingGaussian):

    """
    Represent clustering task on only 2 columns extracted from dimensionality reduction
    """
    
    def get_features(self):
        """
        Extract 2 PCA from preprocessing_hits pipeline
        """
        preprocessed_training=preprocessing_hits.fit_transform(self.training)
        preprocessed_validation=preprocessing_hits.transform(self.validation)
        preprocessed_test=preprocessing_hits.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_bgm(self, reduced_features_valid):
        """
        Perform Bayesian Gaussian Mixture modelling on 2 features columns
        """
        bgm_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('bayesian_gaussian', BayesianGaussianMixture(n_components=10, n_init=10, random_state=2024))])
        bgm_clustering.fit(self.training) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', bgm_clustering.labels_[:5]) # check labels of first 5 training data
        
        y_pred=bgm_clustering.predict(self.validation)
        
        print('The silhouette score obtained as clustering performance measure is:', silhouette_score(reduced_features_valid, y_pred))
        
        return bgm_clustering, y_pred
    
    
    
    def visualize_plot(plot_bgm, bgm_clustering, X_train, size=200):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_bgm(bgm_clustering, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_Gaussian_clustering_result_by_hits"))
        


# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

    actual_clustering, prediction_clusters_valid=clustering_task.perform_bgm(X_valid_features)

    #Columns2Clustering.visualize_plot(Columns2Clustering.plot_bgm, actual_clustering[1], X_train_features)

    #Columns2Clustering.visualize_plot(Columns2Clustering.plot_bgm, actual_clustering[1], X_valid_features)

    
    

import timeit

time_taken = timeit.timeit(lambda: main(), number=2)
print(f"Execution time for gaussian_mixtures_clustering_hits.py is : {time_taken} seconds")
