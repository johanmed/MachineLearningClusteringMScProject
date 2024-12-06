#!/usr/bin/env python
"""
Dependencies: vector_data.py -> data, preprocessing_qtl
Neural networks are used to predict description or trait category of validation data
Modelling by qtl (chromosome number)
"""


# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

from vector_data import X_train, X_valid, X_test, preprocessing_qtl

import numpy as np
import pandas as pd

y_train=X_train['desc']

X_train=X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_valid=X_valid['desc']

X_valid=X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]

y_test=X_test['desc']

X_test=X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf
import matplotlib.pyplot as plt # import plot manager
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

tf.keras.utils.set_random_seed(2024) # set random seed for tf, np and python


out_dir=os.path.abspath('../output/') # define directory to save plots to


from general_deep_learning import MyAnnotationTaskTuning # import MyAnnotationTaskTuning


class Annotation:

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

     
    def extract_features_target_relationship(self, neural_model_sup, y_train, y_valid):
        """
        Find relationships between 2 columns selected (features) and description (target) using neural networks
        Assign to each observation a description to know the type of trait -> supervised learning
        Best model from finetuning used for the purpose
        """
        
        # Convert target variables to one-hot encoded format

        y_train = to_categorical(y_train, num_classes=3)

        y_valid = to_categorical(y_valid, num_classes=3)
        

        neural_assign=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('neural_supervised', neural_model_sup)])
        neural_assign.fit(self.training, y_train, neural_supervised__epochs=10)
        
        y_supervised_pred=neural_assign.predict(self.validation)
        
        print('The classification report is as follows: ', classification_report(y_valid, y_pred_supervised))
        
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
        plt.savefig(os.path.join(out_dir, f"Project_PCA_neural_{type_anno}_annotation_result_by_hits"))

     


# Main

def main():

    annotation_task=Annotation(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=annotation_task.get_features()
    
    if os.path.exists('deep_learning_annotation_qtl/best_annotation_model_by_qtl.keras'):
        
        best_model=tf.keras.models.load_model('deep_learning_annotation_qtl/best_annotation_model_by_qtl.keras')
        
    else:

        hyperband_tuner=kt.Hyperband(MyAnnotationTaskTuning(), objective='val_rmse', seed=2024, max_epochs=10, factor=3, hyperband_iterations=3, overwrite=True, directory='deep_learning_annotation_qtl', project_name='hyperband')
    
        early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=5)
    
        tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S"))
    
        hyperband_tuner.search(X_train_features, y_train, epochs=10, validation_data=(X_valid_features, y_valid), callbacks=[early_stopping_cb, tensorboard_cb])
    
        top3_models=hyperband_tuner.get_best_models(num_models=3)
    
        best_model=top3_models[0]
        
        best_model.save('deep_learning_annotation_qtl/best_annotation_model_by_qtl.keras')

    extracted_annotation=annotation_task.extract_features_target_relationship(best_model, y_train, y_valid)

    Annotation.visualize_plot_annotation(X_valid_features, extracted_annotation, 'predicted')

    Annotation.visualize_plot_annotation(X_valid_features, y_valid, 'actual')




import timeit

time_taken = timeit.timeit(lambda: main(), number=2)
print(f"Execution time for deep_learning_annotation_qtl.py is : {time_taken} seconds")