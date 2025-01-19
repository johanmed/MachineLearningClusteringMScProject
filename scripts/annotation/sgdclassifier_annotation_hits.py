#!/usr/bin/env python

"""
Dependencies: vector_data.py -> data, preprocessing_hits
SGDClassifier is used
Modelling by hits (chromosome number + marker position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import scaled_training_set as X_train
from vector_data import scaled_validation_set as X_valid
from vector_data import scaled_test_set as X_test

from vector_data import preprocessing_hits

import numpy as np
import pandas as pd

y_train=X_train['desc']

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_valid=X_valid['desc']

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_test=X_test['desc']

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set



# 2. Conduct supervised learning to learn how to annotate

import matplotlib.pyplot as plt # import plot manager
from sklearn.linear_model import SGDClassifier # import SGDClassifer for prediction based on DBSCAN clustering
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


out_dir=os.path.abspath('../../output/') # define directory to save plots to


class Annotation:

    """
    Represents annotation operations
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
        
        

     
    def extract_features_target_relationship(X_train, y_train, X_valid, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target)
        Assign to each observation a description to know the type of trait -> supervised learning
        """
        assign_vec=SGDClassifier(random_state=2024)
        assign_vec.fit(X_train, y_train)
        y_supervised_pred=assign_vec.predict(X_valid)
        print(classification_report(y_valid, y_supervised_pred))
        return y_supervised_pred


    def visualize_plot_annotation(X_valid, y_supervised_pred, type_anno):
        """
        Regenerate visualization for clustering adding annotation of description or original trait category to each observation
        Save figure
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_supervised_pred)
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.colorbar(label='Original trait category', spacing='uniform', values=[0, 1, 2])
        plt.savefig(os.path.join(out_dir, f"SGDClassifier_{type_anno}_annotation_result_by_hits"))

     


# Main

def main():

    annotation_task=Annotation(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=annotation_task.get_features()

    extracted_annotation=Annotation.extract_features_target_relationship(X_train_features, y_train, X_valid_features, y_valid)

    Annotation.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Annotation.visualize_plot_annotation(X_valid_features, y_valid, 'actual')




main(0
