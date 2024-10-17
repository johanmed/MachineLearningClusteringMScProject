#!/usr/bin/env python

"""
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies: vector_data.py -> data
KMeans is run twice:
1. Identify the best number of clusters for the data using the training data
2. Proceed to actual training and validation on respective data
RandomForestClassifier is run to predict description or trait category of validation data
Modelling by hits (chromosome number + chromosomal position)
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

from sklearn.cluster import KMeans # import KMeans class
from sklearn.metrics import silhouette_score # import silhouette_score class
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.ensemble import RandomForestClassifier # import RandomForestClassifier
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

    

    def find_best_n_clusters(self, a, b):
        """
        Run KMeans for range of clusters numbers specified by a and b
        """
        best_n_clusters=[] # create a variable that stores the best number of clusters
        sil_score={} # variable storing silhouette score for each number of clusters

        for n_clusters in range(a, b):
            kmeans=KMeans(n_clusters, algorithm='elkan', random_state=2024) # choice of elkan as algorithm made to skip unnecessary calculations and gain ideally in efficiency
            kmeans.fit(self.training)
            #print('The inertia for ', n_clusters, ' clusters is :', kmeans.inertia_) # inertia metrics gives sum of the squared distances between the instances and their closest centroids
            silhouette=silhouette_score(self.training, kmeans.labels_) # silhouette score is believed to be a better metric to select best number of cluster
            sil_score[n_clusters]=silhouette
            #print('The silhouette score for ', n_clusters, ' clusters is :', silhouette)
    
        for i in sil_score.keys():
            if sil_score[i]==max(sil_score.values()):
                best_n_clusters.append([i, sil_score[i]]) # update the best number of clusters

        print('The best number of clusters with its silhouette score is :', best_n_clusters[0])
        
        return best_n_clusters[0][0]


    def perform_kmeans_clustering(self, best_n_clusters):
        """
        Run KMeans for the best number of clusters on training and save predictions and distances to centroids
        """
        kmeans=KMeans(n_clusters=best_n_clusters, algorithm='elkan', random_state=2024)
        kmeans.fit(self.training)
        #print('The labels assigned to the following training data \n', X_train[:5], ' are respectively: \n', kmeans.labels_[:5]) # check labels of first 5 training data
        y_pred=kmeans.predict(self.validation)
        #print('The labels assigned to the following validation data \n', X_valid, ' are respectively: \n', y_pred[:5]) # check labels assigned to first 5 validation data

        distance_inst_centro=kmeans.transform(X_train).round(2) # Save distance of instances to centroids infered for the best number of clusters
        #print('The distances to each centroid for the first 5 instances are: \n', distance_inst_centro[:5]) # can change to see for more
        
        return kmeans, y_pred, distance_inst_centro



    def plot_kmeans(clusterer, X):
        """
        Plot clusters extracted by KMeans
        """

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
                plt.xlabel("Transformed trait category and p-lrt", fontsize=20)
            else:
                plt.tick_params(labelbottom=False)
            if show_ylabels:
                plt.ylabel("Transformed chromosome number and position", fontsize=20, rotation=90)
            else:
                plt.tick_params(labelleft=False)
        

    def visualize_plot(plot_kmeans, clusterer, X_train, index1, index2)
        """
        Generate actual visualization of clusters
        Save figure
        """
        
        plt.figure(figsize=(10, 10))
        plot_kmeans(clusterer, X_train)
        plt.savefig(os.path.join(out_dir, f"Project_KMeans_clustering_RandomForest_prediction_result_by_hits_{index1}_{index2}")))
        plt.show()
        
        
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
        plt.savefig(os.path.join(out_dir, f"Project_KMeans_clustering_RandomForest_{type_anno}_annotation_result_by_hits_{index1}_{index2}"))
        plt.show()



def columns2clustering(index1, index2):
    """
    Perform all clustering operations predefined for the indexes at hand
    """
    clustering_task=Columns2Clustering(X_train, X_valid, X_test, index1, index2)
    datasets=clustering_task.get_all_datasets()
    num_clusters=clustering_task.find_best_n_clusters(10, 1000) # can try out different values
    actual_clustering=clustering_task.perform_kmeans_clustering(num_clusters)
    Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[0], datasets[0], index1, index2)
    prediction_clusters=actual_clustering[1]
    distances_centroids_validation=actual_clustering[2]
    extracted_annotation=Columns2Clustering.extract_features_target_relationship(datasets[0], y_train, datasets[1], y_valid)
    Columns2Clustering.visualize_plot_annotation(datasets[1], extracted_annotation, index1, index2, 'predicted')
    Columns2Clustering.visualize_plot_annotation(datasets[1], y_valid, index1, index2, 'actual')

for i in range(1, 6):
    for j in range(1, 6):
        columns2clustering(i, j)
        


