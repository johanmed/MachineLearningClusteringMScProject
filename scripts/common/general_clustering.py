#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class ModellingBirch:

    """
    Represent ML operations using Birch
    """
    
    def __init__(self, training, validation, test):
        """
        Instantiate a class object
        """
        self.training=training
        self.validation=validation
        self.test=test
        

    def plot_birch(clusterer, X, size, show_xlabels=True, show_ylabels=True):
        """
        Display Birch clustering distinguishing, core, non core and anomalies instances
        Data plotted according to 2 features provided
        """
        new_labels=clusterer.predict(X)
        anomalies_mask=new_labels== -1
        cores=clusterer.subcluster_centers_
        anomalies=X[anomalies_mask]
    
        plt.scatter(X[:, 0], X[:, 1], c=new_labels, marker='o', s=size, cmap='Paired')
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=100, c='b')
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    
        if show_xlabels:
            plt.xlabel("PC 1", fontsize=10)
        else:
            plt.tick_params(labelbottom=False)
    
        if show_ylabels:
            plt.ylabel("PC 2", fontsize=10, rotation=90)
        else:
            plt.tick_params(labelleft=False)
    
    

class ModellingKMeans:

    """
    Represent ML operations using KMeans
    """
    
    def __init__(self, training, validation, test):
        """
        Instantiate a class object
        """
        self.training=training
        self.validation=validation
        self.test=test
        
        
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
        plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k', alpha=0.5)
    
    
    
        def plot_data(X):
            """
            Plot data according 2 columns selected
            """
            plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
        
        plot_data(X)
    
    
    
        def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

            """
            Represent centroids differently
            """
            if weights is not None:
                centroids=centroids[weights > weights.max()/10]
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8, color=circle_color, zorder=10, alpha=0.4)
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=0.6)
    

        if show_centroids:
            plot_centroids(clusterer.cluster_centers_)
        
        if show_xlabels:
            plt.xlabel("PC 1", fontsize=20)
        else:
            plt.tick_params(labelbottom=False)
        if show_ylabels:
            plt.ylabel("PC 2", fontsize=20, rotation=90)
        else:
            plt.tick_params(labelleft=False)




class ModellingGaussian:

    """
    Represent ML operations using Gaussian Mixture
    """
    
    def __init__(self, training, validation, test):
        """
        Instantiate a class object
        """
        self.training=training
        self.validation=validation
        self.test=test
        
    def plot_bgm(clusterer, X, resolution=1000, show_xlabels=True, show_ylabels=True):
        mins = X.min(axis=0) - 0.1
        maxs = X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
        Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12))
        plt.contour(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12), linewidths=1, colors='k')

        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, linewidths=2, colors='r', linestyles='dashed')
        
        plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
        
        def plot_centroids(centroids, weights, circle_color='w', cross_color='k'):

            """
            Represent centroids differently
            """
            if weights is not None:
                centroids=centroids[weights > weights.max()/10]
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8, color=circle_color, zorder=10, alpha=0.4)
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=0.6)
    
    
        plot_centroids(clusterer.means_, clusterer.weights_)

        if show_xlabels:
            plt.xlabel("PC 1", fontsize=20)
        else:
            plt.tick_params(labelbottom=False)
            
        if show_ylabels:
            plt.ylabel("PC 2", fontsize=20, rotation=90)
        else:
            plt.tick_params(labelleft=False)
