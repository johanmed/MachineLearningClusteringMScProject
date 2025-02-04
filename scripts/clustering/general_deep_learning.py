#!/usr/bin/env python

"""
This script contains deep learning finetuning classes for clustering and annotation
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt


class MyClusteringTaskTuning(kt.HyperModel):
    """
    Represents hyperparameter tuning for clustering task
    """
    def build(self, hp):
        """
        Finetuning at the model level
        """
        n_hidden=hp.Int('n_hidden', min_value=1, max_value=50, default=10)
        n_neurons=hp.Int('n_neurons', min_value=10, max_value=100)
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')
        optimizer=hp.Choice('optimizer', values=['sgd', 'adam'])
        hidden_activation_func=hp.Choice('hidden_activation_func', values=['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish'])
        
        if optimizer=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        
        hidden_layers_dict={}
        for h in range(1, n_hidden+1):
            hidden_layers_dict[h]=tf.keras.layers.Dense(units=n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func)
            
        concat_layer=tf.keras.layers.Concatenate()
        output_layer=tf.keras.layers.Dense(units=3, activation='softmax') # 3 clusters found by both birch hits and birch qtl
        
        layers={}
        layers['input_unsup']=tf.keras.layers.Input(shape=(2,))
        
        for k in hidden_layers_dict:
            layers[f'hidden{k}']=hidden_layers_dict[k](layers[list(layers.keys())[-1]]) # layer output passed to hidden is the previous one before the 2 above just added
            
        layers['concatenated']=concat_layer([layers['input_unsup'], layers[list(layers.keys())[-1]]]) # last layer of the dictionary passed with the input layer for concatenation
        layers['output_unsup']=output_layer(layers['concatenated'])
        
        neural_model_unsup=tf.keras.Model(inputs=[layers['input_unsup']], outputs=[layers['output_unsup']])
        
        neural_model_unsup.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return neural_model_unsup
        
        
    def fit(self, hp, neural_model_unsup, X, y, **kwargs):
        """
        Finetuning of second preprocessing (normalization) and fit parameters
        """
        batch_size=hp.Int('batch_size', min_value=10, max_value=2000)
        
        if hp.Boolean('normalize'):
            normalization_layer=tf.keras.layers.Normalization()
            X=normalization_layer(X)
        
        return neural_model_unsup.fit(X, y, batch_size=batch_size, **kwargs)



class MyAnnotationTaskTuning(kt.HyperModel):
    """
    Represents hyperparameter tuning for annotation task
    """
    def build(self, hp):
        """
        Finetuning at the model level
        """
        n_hidden=hp.Int('n_hidden', min_value=5, max_value=50)
        n_neurons=hp.Int('n_neurons', min_value=10, max_value=100)
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')
        optimizer=hp.Choice('optimizer', values=['sgd', 'adam'])
        hidden_activation_func=hp.Choice('activation_func', values=['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish'])
        
        if optimizer=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        
        hidden_layers_dict={}
        for h in range(1, n_hidden+1):
            hidden_layers_dict[h]=tf.keras.layers.Dense(n_neurons, kernel_initializer='he_normal', activation=hidden_activation_func)
            
        concat_layer=tf.keras.layers.Concatenate()
        output_layer=tf.keras.layers.Dense(units=3, activation='softmax')
        
        layers={}
        layers['input_sup']=tf.keras.layers.Input(shape=(2,))
        
        for k in hidden_layers_dict:
            layers[f'hidden{k}']=hidden_layers_dict[k](layers[list(layers.keys())[-1]]) # layer output passed to hidden is the previous one before the 2 above just added
            
            
        layers['concatenated']=concat_layer([layers['input_sup'], layers[list(layers.keys())[-1]]]) # last layer of the dictionary passed with the input layer for concatenation
        layers['output_sup']=output_layer(layers['concatenated'])
        
        neural_model_sup=tf.keras.Model(inputs=[layers['input_sup']], outputs=[layers['output_sup']])
        
        neural_model_sup.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return neural_model_sup


    def fit(self, hp, neural_model_sup, X, y, **kwargs):
        """
        Finetuning of second preprocessing (normalization) and fit parameters
        """
        batch_size=hp.Int('batch_size', min_value=10, max_value=2000)
        
        if hp.Boolean('normalize'):
            normalization_layer=tf.keras.layers.Normalization()
            X=normalization_layer(X)
        
        return neural_model_sup.fit(X, y, batch_size=batch_size, **kwargs)


