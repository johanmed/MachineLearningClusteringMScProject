#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies: vector_data.py -> data
DBSCAN is run on training data to get clustering
SupportVectorMachine is run on clusters extracted by DBSCAN to predict clusters
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format
from vector_data import X_train, X_valid, X_test

import numpy as np

X_train=np.array(X_train.iloc[: 5000, [0, 2]]) # select transformed_pos (2) and transformed_desc (0) columns
X_valid=np.array(X_valid.iloc[: 5000, [0, 2]]) # same
X_test=np.array(X_test.iloc[: 5000, [0, 2]]) # same



# 2. Proceed to DBSCAN clustering
from sklearn.cluster import DBSCAN # import DBSCAN class for clustering


dbscan=DBSCAN(eps=0.25)
dbscan.fit(X_train) # work with transformed_pos and transformed_desc columns only
print('The labels for the first 5 training data are: ', dbscan.labels_[:5]) # check labels of first 5 training data



# 3. Plot DBSCAN clustering on training data for visualization


# 3.1. Define function for plotting

import matplotlib.pyplot as plt # import plot manager

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    """
    Display DBSCAN clustering distinguishing, core, non core and anomalies instances
    Data plotted according to transformed_pos (1) and transformed_desc (-1)
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
        plt.xlabel("Trait category", fontsize=10)
    else:
        plt.tick_params(labelbottom=False)
    
    if show_ylabels:
        plt.ylabel("Scaled chromosomal position", fontsize=10, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    


# 3.2. Proceed to plotting

import os

plt.figure(figsize=(10, 10))
plot_dbscan(dbscan, X_train, size=500)
out_dir=os.path.abspath('../output/')
plt.savefig(os.path.join(out_dir, " Project DBSCAN clustering and SVM prediction training result"))
plt.show()


# 4. Run SVC on DBSCAN components and labels for prediction

from sklearn.svm import SVC # import SVC for prediction based on DBSCAN clustering

sup_vec=SVC(kernel='rbf')
print(len(X_train))
print(len(dbscan.components_))
#print(sum(dbscan.labels_[dbscan.core_sample_indices_]))
sup_vec.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train Gaussian RBF SVC on data and labels extracted from DBSCAN

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
y_pred=sup_vec.predict(X_valid) # get labels for validation data based on nearest cluster
y_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data



# 5. Save distance of instances to clusters

print('The distances to each cluster for the first 5 instances are: ', y_dist[:5].round(2)) # can change to see more




