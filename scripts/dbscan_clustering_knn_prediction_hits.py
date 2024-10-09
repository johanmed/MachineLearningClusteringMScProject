#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies: vector_data.py -> data
DBSCAN is run on training data to get clustering
KNeighborsClassifier is run on clusters extracted by DBSCAN to predict clusters
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format
from vector_data import X_train, X_valid, X_test

import numpy as np

X_train=X_train[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # select 5000 first columns of combined and transformed columns

X_valid=X_valid[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same

X_test=X_test[['transformed_combined_chr_num_pos1', 'transformed_combined_chr_num_pos2', 'transformed_combined_chr_num_pos3', 'transformed_combined_chr_num_pos4', 'transformed_combined_chr_num_pos5', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same




# 2. Select 2 columns to make clustering on

class Clustering:

    """
    Represents a clustering task between 2 complementary features
    """
    def __init__(self, training, validation, test):
        self.training=training
        self.validation=validation
        self.test=test
    
    def get_features_all_datasets(self, feature1, feature2):
        return np.array(self.training[[feature1, feature2]]), np.array(self.validation[[feature1, feature2]]), np.array(self.test[[feature1, feature2]])


print('It is time to select the features you want to do clustering on!')

feature1=input('Enter your first feature: ')
feature2=input('Enter your second feature: ')
clustering_task1=Clustering(X_train, X_valid, X_test)
X_train, X_valid, X_test=clustering_task1.get_features_all_datasets(feature1, feature2)


# 3. Proceed to DBSCAN clustering
from sklearn.cluster import DBSCAN # import DBSCAN class for clustering


dbscan=DBSCAN(eps=0.25)
dbscan.fit(X_train) # work with 2 features provided
print('The labels for the first 5 training data are: ', dbscan.labels_[:5]) # check labels of first 5 training data



# 4. Plot DBSCAN clustering on training data for visualization


# 4.1. Define function for plotting

import matplotlib.pyplot as plt # import plot manager

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
        plt.xlabel("Trait category", fontsize=20)
    else:
        plt.tick_params(labelbottom=False)
    
    if show_ylabels:
        plt.ylabel("Scaled chromosomal position", fontsize=20, rotation=90)
    else:
        plt.tick_params(labelleft=False)
    


# 4.2. Proceed to plotting

import os

plt.figure(figsize=(10, 10))
plot_dbscan(dbscan, X_train, size=500)
out_dir=os.path.abspath('../output/')
plt.savefig(os.path.join(out_dir, "Project Improved DBSCAN clustering and KNN prediction training result"))
plt.show()



# 5. Run KNeighborsClassfiers on DBSCAN components and labels for prediction

from sklearn.neighbors import KNeighborsClassifier # import KNeighborsClassifier for prediction based on DBSCAN clustering

knn=KNeighborsClassifier()
print(len(X_train))
print(len(dbscan.components_)) # expect that this length is less than the previous
print(len(dbscan.labels_))
print(len(dbscan.labels_[dbscan.core_sample_indices_])) # expect that this length is also less than the previous
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train knn on data and labels extracted from DBSCAN
y_dist, y_pred_id=knn.kneighbors(X_valid) # get distances and indices to nearest clusters obtained for training data for validation data
y_pred=dbscan.labels_[dbscan.core_sample_indices_][y_pred_id] # get labels for validation data based on nearest cluster
y_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
#print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data



# 5. Save distance of instances to clusters

print('The distances to each cluster for the first 5 instances are: ', y_dist[:5].round(2)) # can change to see more



