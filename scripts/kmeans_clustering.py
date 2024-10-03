#!/usr/bin/env python

"""
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies: vector_data.py -> data
KMeans is run twice:
1. Identify the best number of clusters for the data using the training data
2. Proceed to actual training and validation on respective data

Save distance of each instance to centroid
Plot clustering of training data for visualization
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test

import numpy as np

X_train=np.array(X_train.iloc[: 5000, [0, 2]]) # select transformed_pos (1) and transformed_desc (-1) columns
X_valid=np.array(X_valid.iloc[: 5000, [0, 2]]) # same
X_test=np.array(X_test.iloc[: 5000, [0, 2]]) # same


# 2. Proceed to KMeans clustering

from sklearn.cluster import KMeans # import KMeans class
from sklearn.metrics import silhouette_score # import silhouette_score class



# 2.1. Run KMeans for different numbers of cluster on training set and select the best number of clusters


# 2.1.1. Define function to run KMeans and select the best number of clusters

best_n_clusters=[] # create a variable that stores the best number of clusters

def find_n_clusters(a, b):
    """
    Run KMeans for range of number clusters specified by a and b
    """
    sil_score={} # variable storing silhouette score for each number of clusters

    ran=range(a, b) # range of number of clusters to use for run
    for n_clusters in ran:
        kmeans=KMeans(n_clusters, algorithm='elkan', random_state=2024) # choice of elkan as algorithm made to skip unnecessary calculations and gain ideally in efficiency
        kmeans.fit(X_train)
        #print('The inertia for ', n_clusters, ' clusters is :', kmeans.inertia_) # inertia metrics gives sum of the squared distances between the instances and their closest centroids
        silhouette=silhouette_score(X_train, kmeans.labels_) # silhouette score is believed to be a better metric to select best number of cluster
        sil_score[n_clusters]=silhouette
        #print('The silhouette score for ', n_clusters, ' clusters is :', silhouette)
    
    for i in sil_score.keys():
        if sil_score[i]==max(sil_score.values()):
            best_n_clusters.append([i, sil_score[i]]) # update the best number of clusters

    print('The best number of clusters with its silhouette score in that range ', ran, ' is :', best_n_clusters[0])

# 2.1.2. Find the best number of clusters

find_n_clusters(1500, 1700) # can experiment different values but experimentations show that the silhouette score of this model is less than 0.5 when the number of clusters is less than 1000

# 2.2. Run KMeans for the best number of clusters on training and display learning results

kmeans=KMeans(n_clusters=best_n_clusters[0][0], algorithm='elkan', random_state=2024)
kmeans.fit(X_train)
#print('The labels assigned to the following training data \n', X_train[:5], ' are respectively: \n', kmeans.labels_[:5]) # check labels of first 5 training data
y_pred=kmeans.predict(X_train)
#print('The labels assigned to the following validation data \n', X_valid, ' are respectively: \n', y_pred[:5]) # check labels assigned to first 5 validation data


# 3. Save distance of instances to centroids infered for the best number of clusters

distance_inst_centro= kmeans.transform(X_train).round(2)
print('The distances to each centroid for the first 5 instances are: \n', distance_inst_centro[:5]) # can change to see for more



# 4. Plot clustering on training data for visualization

# 4.1. Define functions for plotting data using KMeans

import matplotlib.pyplot as plt # import plot manager

def plot_data(X):
    """
    plot data according to transformed_pos (1) and transformed_desc (-1)
    """
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

    """
    Represent centroids differently
    """
    if weights is not None:
        centroids=centroids[weights > weights.max()/10]
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8, color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)
    

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    """
    Display the clustering of the data, the centroids and the decision boundaries of kmeans
    """
    mins=X.min(axis=0) - 0.1
    maxs=X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    Z=clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z=Z.reshape(xx.shape)
    
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    
    plot_data(X)
    
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
        
    if show_xlabels:
        plt.xlabel("Trait category", fontsize=10)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("Scaled chromosomal position", fontsize=10, rotation=0)
    else:
        plt.tick_params(labelleft=False)
        
# 4.2. Proceed to plotting

import os

plt.figure(figsize=(10, 10))
plot_decision_boundaries(kmeans, X_train)
out_dir=os.path.abspath('../output/')
plt.savefig(os.path.join(out_dir, "Project KMeans clustering training result"))
plt.show()
