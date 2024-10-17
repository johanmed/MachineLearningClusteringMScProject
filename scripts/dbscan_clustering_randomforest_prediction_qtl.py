#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies: vector_data.py -> data
DBSCAN is run on training data to get clustering
RandomForestClassifier is run on clusters extracted by DBSCAN to predict clustering and description or trait category of validation data
Modelling by QTL peaks (chromosome number)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test

import numpy as np

y_train=X_train['desc'][:5000]

X_train=X_train[['transformed_chr_num', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # select 5000 first columns of combined and transformed columns

y_valid=X_valid['desc'][:5000]

X_valid=X_valid[['transformed_chr_num', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same

y_test=X_test['desc'][:5000]

X_test=X_test[['transformed_chr_num', 'transformed_combined_desc_p_lrt1', 'transformed_combined_desc_p_lrt2', 'transformed_combined_desc_p_lrt3', 'transformed_combined_desc_p_lrt4', 'transformed_combined_desc_p_lrt5']][:5000] # same




# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import DBSCAN # import DBSCAN class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.ensemble import RandomForestClassifier # import RandomForestClassifier for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report


out_dir=os.path.abspath('../output/') # define directory to save plots to


class Columns2Clustering:

    """
    Represent clustering task on 2 specific features columns
    """
    
    def __init__(self, training, validation, test, index):
        """
        Instantiate a class object
        """
        self.training=training
        self.validation=validation
        self.test=test
        self.index=index
        
        
    def get_all_datasets(self):
        """
        Return new datasets with only the specific features columns selected
        """
        return np.array(self.training[[f'transformed_chr_num', f'transformed_combined_desc_p_lrt{self.index}']]), np.array(self.validation[[f'transformed_chr_num', f'transformed_combined_desc_p_lrt{self.index}']]), np.array(self.test[[f'transformed_chr_num', f'transformed_combined_desc_p_lrt{self.index}']])


    def perform_dbscan_clustering(self):
        """
        Perform DBSCAN clustering on 2 features columns
        """
        dbscan=DBSCAN(eps=0.25)
        dbscan.fit(np.array(self.training[[f'transformed_chr_num', f'transformed_combined_desc_p_lrt{self.index}']])) # work with 2 features provided
        #print('The labels for the first 5 training data are: ', dbscan.labels_[:5]) # check labels of first 5 training data
        return dbscan
    
    
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
            plt.xlabel("Transformed trait category and p-lrt", fontsize=10)
        else:
            plt.tick_params(labelbottom=False)
    
        if show_ylabels:
            plt.ylabel("Transformed chromosome number", fontsize=10, rotation=90)
        else:
            plt.tick_params(labelleft=False)
    
    
    
    def visualize_plot(plot_dbscan, dbscan, X_train, index, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plot_dbscan(dbscan, X_train, size)
        plt.savefig(os.path.join(out_dir, f"Project_DBSCAN_clustering_RandomForest_prediction_result_by_qtl_{index}"))
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


    def visualize_plot_annotation(X_valid, y_supervised_pred, index, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_supervised_pred)
        plt.xlabel("Transformed trait category and p-lrt", fontsize=10)
        plt.ylabel("Transformed chromosome number", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_DBSCAN_clustering_RandomForest_{type_anno}_annotation_result_by_qtl_{index}"))
        plt.show()

     
     
     
def columns2clustering(index):
    """
    Perform all clustering operations predefined for the index at hand
    """
    clustering_task=Columns2Clustering(X_train, X_valid, X_test, index)
    datasets=clustering_task.get_all_datasets()
    actual_clustering=clustering_task.perform_dbscan_clustering()
    Columns2Clustering.visualize_plot(Columns2Clustering.plot_dbscan, actual_clustering, datasets[0], index)
    prediction_clusters=Columns2Clustering.predict_dbscan_clustering(actual_clustering, datasets[1])
    extracted_annotation=Columns2Clustering.extract_features_target_relationship(datasets[0], y_train, datasets[1], y_valid)
    Columns2Clustering.visualize_plot_annotation(datasets[1], extracted_annotation, index, 'predicted')
    Columns2Clustering.visualize_plot_annotation(datasets[1], y_valid, index, 'actual')


for i in range(1, 6):
    columns2clustering(i)
        


















































#!/usr/bin/env python

"""
Summary:
This script contains code to run DBSCAN clustering algorithm on data
Dependencies: vector_data.py -> data
DBSCAN is run on training data to get clustering
RandomForest is run on clusters extracted by DBSCAN to predict clusters
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format
from vector_data import X_train, X_valid, X_test

import numpy as np

X_train=np.array(X_train.iloc[: 5000, [0, 2, -1]]) # select desc, transformed_pos and transformed_desc columns
X_valid=np.array(X_valid.iloc[: 5000, [0, 2, -1]]) # same
X_test=np.array(X_test.iloc[: 5000, [0, 2, -1]]) # same



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
    Data plotted according to desc transformed_pos
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
    


# 3.2. Proceed to plotting

import os

plt.figure(figsize=(10, 10))
plot_dbscan(dbscan, X_train, size=500)
out_dir=os.path.abspath('../output/')
plt.savefig(os.path.join("Project DBSCAN clustering and Random Forest prediction training result"))
plt.show()


# 4. Run Random Forest on DBSCAN components and labels for prediction

from sklearn.ensemble import RandomForestClassifier # import RandomForestClassifier for prediction based on DBSCAN clustering

random_forest=RandomForestClassifier(random_state=2024)
print(len(X_train))
print(len(dbscan.components_))
random_forest.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]) # train Random Forest model on data and labels extracted from DBSCAN

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
y_pred=random_forest.predict(X_valid) # get labels for validation data based on nearest cluster
y_pred[y_dist>0.2]=-1 # detect anomalies by setting maximum distance allowed between instance and nearest cluster to 0.2 (can be changed)
print('The labels for the first 5 validation data are: \n', y_pred[:5]) # check labels for the first 5 validation data



# 5. Save distance of instances to clusters

print('The distances to each cluster for the first 5 instances are: ', y_dist[:5].round(2)) # can change to see more




