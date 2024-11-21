#!/usr/bin/env python
"""
Summary:
This script contains code to do clustering using neural networks (deep learning)
Dependencies: 
- vector_data.py -> data, preprocessing_qtl
Neural networks are used to get clustering and to predict description or trait category of validation data
Modelling by hits (chromosome number + marker position)
"""


# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_qtl

import numpy as np

y_train=X_train['desc']

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]

y_valid=X_valid['desc']

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]

y_test=X_test['desc']

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]



# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf # import DBSCAN class for clustering
import matplotlib.pyplot as plt # import plot manager
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(2024) # set random seed for tf, np and python


out_dir=os.path.abspath('../output/') # define directory to save plots to



class Columns2Clustering:

    """
    Represent clustering task on 2 columns extracted from dimensionality reduction
    """
    
    def __init__(self, training, validation, test):
        self.training=training
        self.validation=validation
        self.test=test
    
    
    def get_features(self):
        """
        Extract 2 PCA from preprocessing_hits pipeline
        """
        preprocessed_training=preprocessing_qtl.fit_transform(self.training)
        preprocessed_validation=preprocessing_qtl.transform(self.validation)
        preprocessed_test=preprocessing_qtl.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_neural_clustering(self, reduced_features_train):
        """
        Perform neural clustering on 2 features columns
        """
        normalization_layer_unsup=tf.keras.layers.Normalization(input_shape=(2,))
        neural_model_unsup=tf.keras.models.Sequential([
        normalization_layer_unsup,
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=5, activation='softmax'), # number of clusters set to 5 here
        ])
        neural_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('neural_unsupervised', neural_model_unsup)])
        neural_clustering[1].compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        normalization_layer_unsup.adapt(np.array(reduced_features_train))
        neural_clustering.fit(self.training, y_train, neural_unsupervised__epochs=10, neural_unsupervised__batch_size=1000)
        #print('The labels for the first 5 training data are: ', dbscan_clustering.labels_[:5]) # check labels of first 5 training data

        return neural_clustering
    
    
    
    def visualize_plot(neural_clustering, X_train, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        y_pred_unsup_train=neural_clustering.predict(X_train)
        
        clusters_unsup_train=[]
        for i in y_pred_unsup_train:
            temp=[]
            for (x,y) in enumerate(i):
                if y==max(i):
                    temp.append(x)
            clusters_unsup_train.append(temp[0])
        
        plt.figure(figsize=(10, 10))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=clusters_unsup_train)
        plt.xlabel("PCA 1", fontsize=10)
        plt.ylabel("PCA 2", fontsize=10, rotation=90)
        plt.savefig(os.path.join(out_dir, f"Project_PCA_neural_clustering_result_by_qtl"))
        plt.show()
        
        
     
    def predict_neural_clustering(neural_clustering, X_valid):
        """
        Use neural networks to predict clustering on validation set
        """
        y_pred_unsup_valid=neural_clustering.predict(X_valid)
        
        #print('The labels for the first 5 validation data are: \n', y_pred_unsup_valid[:5]) # check labels for the first 5 validation data
        
        return y_pred_unsup_valid


     
    def extract_features_target_relationship(self, reduced_features_train, y_train, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target) using neural networks
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        normalization_layer_sup=tf.keras.layers.Normalization(input_shape=(2,))
        neural_model_sup=tf.keras.models.Sequential([
        normalization_layer_sup,
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(1),
        ])
        
        
        # Convert target variables to one-hot encoded format

        y_train = to_categorical(y_train, num_classes=3)

        y_valid = to_categorical(y_valid, num_classes=3)



        neural_assign=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('neural_supervised', neural_model_sup)])
        neural_assign[1].compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])
        normalization_layer_sup.adapt(np.array(reduced_features_train))
        neural_assign.fit(self.training, y_train, neural_supervised__epochs=10, neural_supervised__batch_size=1000)
        y_supervised_pred=neural_assign.predict(self.validation)
        
        #print('The prediction for the first 5 validation data is :', y_pred[:5])
        
        return y_supervised_pred


    def visualize_plot_annotation(X_valid, y_supervised_pred, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_supervised_pred)
        plt.xlabel("PCA 1", fontsize=10)
        plt.ylabel("PCA 2", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"Project_PCA_neural_clustering_{type_anno}_annotation_result_by_qtl"))
        plt.show()

     


# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

    actual_clustering=clustering_task.perform_neural_clustering(X_train_features)

    Columns2Clustering.visualize_plot(actual_clustering[1], X_train_features)

    Columns2Clustering.visualize_plot(actual_clustering[1], X_valid_features)

    prediction_clusters=Columns2Clustering.predict_neural_clustering(actual_clustering[1], X_valid_features)

    extracted_annotation=clustering_task.extract_features_target_relationship(X_train_features, y_train, y_valid)

    Columns2Clustering.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Columns2Clustering.visualize_plot_annotation(X_valid_features, y_valid, 'actual')



import timeit

time_taken = timeit.timeit(lambda: main(), number=10)
print(f"Execution time for deep_learning_clustering_qtl.py is : {time_taken} seconds")
