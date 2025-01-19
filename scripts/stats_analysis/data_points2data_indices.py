#!/usr/bin/env python

"""
This script uses best clustering model to extract data index and cluster assignment for the whole dataset
"""

import os
import tensorflow as tf

from vector_data import processed_X, preprocessing_qtl

y_full=processed_X['desc']

X_full=processed_X[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num']]


class NewColumns2Clustering:

    """
    Represent clustering task on 2 columns extracted from dimensionality reduction
    """
    
    def __init__(self, data):
        self.data=data # use whole dataset

    def get_features(self):
        """
        Extract 2 PCA from preprocessing_qtl pipeline
        """
        preprocessed_data=preprocessing_qtl.fit_transform(self.data)
        
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
        
    def predict_neural_clustering(neural_clustering, data, get_clusters_labels):
        """
        Use neural networks to predict clustering on whole set
        """
        y_pred_unsup_valid=neural_clustering.predict(data)
        
        clusters_unsup_valid=get_clusters_labels(y_pred_unsup_valid)
        
        return clusters_unsup_valid


def main():

    clustering_task=NewColumns2Clustering(X_full)

    X_full_features=clustering_task.get_features() # get preprocessed features for whole dataset
    
    if os.path.exists('../clustering/deep_learning_clustering_qtl/best_clustering_model_by_qtl.keras'):
        
        best_model=tf.keras.models.load_model('../clustering/deep_learning_clustering_qtl/best_clustering_model_by_qtl.keras')
        
        prediction_clusters=NewColumns2Clustering.predict_neural_clustering(best_model, X_full_features, NewColumns2Clustering.get_clusters_labels)
        
        container=[]
        for (i, j) in enumerate(prediction_clusters):
            container.append([i, j]) # save data index and cluster assigned
            
        to_save=pd.DataFrame(container)
        
        num=int(input('Which chunk number are you analyzing?: '))
        
        to_save.to_csv(f'../../../../data_indices/data_indices_clusters{num}.csv', index=False, header=False)
            
    else:
        print('Best model not found, sorry :)')
    


main()
