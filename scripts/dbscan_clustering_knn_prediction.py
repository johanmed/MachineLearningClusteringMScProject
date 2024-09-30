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

X_train=np.array(X_train.iloc[:, [1,-1]]) # select transformed_pos (1) and transformed_p_lrt (-1) columns
X_valid=np.array(X_valid.iloc[:, [1,-1]]) # same
X_test=np.array(X_test.iloc[:, [1,-1]]) # same



# 2. Proceed to DBSCAN clustering
from sklearn.cluster import DBSCAN # import DBSCAN class for clustering


dbscan=DBSCAN()
dbscan.fit(X_train) # work with transformed_pos and transformed_p_lrt columns only
#print('The labels for the first 5 training data are: ', dbscan.labels_[:5]) # check labels of first 5 training data



# 3. Plot DBSCAN clustering on training data for visualization


# 3.1. Define function for plotting

import matplotlib.pyplot as plt # import plot manager

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    """
    Display DBSCAN clustering distinguishing, core, non core and anomalies instances
    Data plotted according to transformed_pos (1) and transformed_p_lrt (-1)
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
        plt.xlabel("Scaled chromosomal position", fontsize=10)
    else:
        plt.tick_params(labelbottom=False)
    
    if show_ylabels:
        plt.ylabel("Scaled p-value", fontsize=10, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    


# 3.2. Proceed to plotting

#plt.figure(figsize=(10, 10))
#plot_dbscan(dbscan, X_train, size=500)
#plt.savefig("DBSCAN clustering and KNN prediction training result")
#plt.show()



# 4. Run KNeighborsClassfiers on DBSCAN components and labels for prediction

from sklearn.neighbors import KNeighborsClassifier # import KNeighborsClassifier for prediction based on DBSCAN clustering

knn=KNeighborsClassifier()
#print(len(X_train))
#print(len(dbscan.components_)) # expect that this length is less than the previous
#print(len(dbscan.labels_))
#print(len(dbscan.labels_[dbscan.core_sample_indices_])) # expect that this length is also less than the previous
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train knn on data and labels extracted from DBSCAN
y_dist, y_pred_id=knn.kneighbors(X_valid) # get distances and indices to nearest clusters obtained for training data for validation data
y_pred=dbscan.labels_[dbscan.core_sample_indices_][y_pred_id] # get labels for validation data based on nearest cluster
y_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
#print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data



# 5. Save distance of instances to clusters

print('The distances to each cluster for the first 5 instances are: ', y_dist[:5].round(2)) # can change to see more



