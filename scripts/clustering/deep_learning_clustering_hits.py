#!/usr/bin/env python
"""
Summary:
This script contains code to do clustering using neural networks (deep learning)
Dependencies: 
- vector_data.py -> data, preprocessing_hits
Neural networks are used to get clustering and to predict description or trait category of validation data
Modelling by hits (chromosome number + marker position)
"""


# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

import os

from vector_data_post import scaled_training_set as X_train
from vector_data_post import scaled_validation_set as X_valid
from vector_data_post import scaled_test_set as X_test

from vector_data_pre import preprocessing_hits

import numpy as np
import pandas as pd

y_train=X_train['y_hits']

desc_train=X_train['desc']

X_train=X_train[['p_lrt', 'chr_num', 'pos']]

y_valid=X_valid['y_hits']

desc_valid=X_valid['desc']

X_valid=X_valid[['p_lrt', 'chr_num', 'pos']]

y_test=X_test['y_hits']

desc_test=X_test['desc']

X_test=X_test[['p_lrt', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set

y_train_full=pd.concat([y_train, y_valid])

# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf
import matplotlib.pyplot as plt # import plot manager
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from random import choice
from pathlib import Path
from time import strftime
import keras_tuner as kt

from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(2024) # set random seed for tf, np and python

out_dir=os.path.abspath('../../output/') # define directory to save plots to


from general_deep_learning import MyClusteringTaskTuning # import MyClusteringTaskTuning


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
        preprocessed_training=preprocessing_hits.fit_transform(self.training)
        preprocessed_validation=preprocessing_hits.transform(self.validation)
        preprocessed_test=preprocessing_hits.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_neural_clustering(self, best_clustering_model, X_valid_features, y_train, y_valid):
        """
        Perform neural clustering on 2 features columns using best model extracted from tuning
        """
        neural_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('best_clustering_model', best_clustering_model)])
        neural_clustering.fit(self.training, y_train, best_clustering_model__validation_data=(X_valid_features, y_valid))
        
        return neural_clustering
    
    
    
    def get_clusters_labels(raw_predictions_proba):
    
        """
        Loop through output dimensions and select the cluster with the highest probability
        """
        clusters_unsup=[]
        for i in raw_predictions_proba:
            temp=[]
            for (x,y) in enumerate(i):
                if y==max(i):
                    temp.append(x)
            clusters_unsup.append(choice(temp))
        
        return clusters_unsup
    
    
    
    def visualize_plot(neural_clustering, X, get_clusters_labels, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        y_pred_unsup_train=neural_clustering.predict(X)
                
        clusters_unsup_train=get_clusters_labels(y_pred_unsup_train)
        
        #print('The silhouette score obtained as clustering performance measure on training set is:', silhouette_score(X_train, clusters_unsup_train))
        
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=clusters_unsup_train)
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.savefig(os.path.join(out_dir, f"Deep_learning_clustering_result_by_hits"), dpi=500)
        
        
        
    def annotate_plot(neural_clustering, X, desc, size=500):
        """
        Annotate plot with trait categories
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=list(desc))
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.savefig(os.path.join(out_dir, f"Deep_learning_clustering_annotated_result_by_hits"), dpi=500)
        
     
    def predict_neural_clustering(neural_clustering, X_valid, get_clusters_labels):
        """
        Use neural networks to predict clustering on validation set
        """
        y_pred_unsup_valid=neural_clustering.predict(X_valid)
        
        clusters_unsup_valid=get_clusters_labels(y_pred_unsup_valid)
        
        
        return clusters_unsup_valid
     


# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()
    
    if os.path.exists('deep_learning_clustering_hits/best_checkpoint.keras'):
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_clustering_hits/best_checkpoint.keras')

    else:
    
        hyperband_tuner=kt.Hyperband(MyClusteringTaskTuning(), objective='val_accuracy', seed=2024, max_epochs=10, factor=2, hyperband_iterations=2, overwrite=True, directory='deep_learning_clustering_hits', project_name='hyperband')
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('deep_learning_clustering_hits/best_checkpoint.keras', save_best_only=True)

        early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2)
    
        tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S"))
    
        hyperband_tuner.search(X_train_features, y_train, epochs=10, validation_data=(X_valid_features, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    
        top3_models=hyperband_tuner.get_best_models(num_models=3)
    
        best_model=top3_models[0]
        
        best_model.save('deep_learning_clustering_hits/best_clustering_model_by_hits.keras')
        
        
        
        
    actual_clustering=clustering_task.perform_neural_clustering(best_model, X_valid_features, y_train, y_valid)

    #Columns2Clustering.visualize_plot(actual_clustering[1], X_train_features, Columns2Clustering.get_clusters_labels)

    Columns2Clustering.visualize_plot(actual_clustering[1], X_valid_features, Columns2Clustering.get_clusters_labels)
    
    Columns2Clustering.annotate_plot(actual_clustering[1], X_valid_features, desc_valid)

    prediction_clusters=Columns2Clustering.predict_neural_clustering(actual_clustering[1], X_valid_features, Columns2Clustering.get_clusters_labels)



main()
