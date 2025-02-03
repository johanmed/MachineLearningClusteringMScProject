#!/usr/bin/env python

"""
Summary:
This file contains code to load data from dataset AFTER unsupervised machine learning
Data of each column are plotted in histogram to assess quality
Training, validation and test sets are defined from X data randomly
Features of the training set are scaled and validation and test set are transformed accordingly
Data of each column of the training are plotted in histogram to confirm quality
"""


# 1. Read data by chunks

import numpy as np
import pandas as pd

from generalize_cluster_assignment import full_X_rec as full_X

print('full_X looks like: \n', full_X)


# 2. Define training, validation and test sets

from sklearn.model_selection import train_test_split # import utility for splitting

from vector_data_pre import define_sets
    
training_set, validation_set, test_set=define_sets(full_X)


training_validation_set=pd.concat([training_set, validation_set]) # get training + validation set


# 3. Plot histogram of training features and assess quality

import matplotlib.pyplot as plt # import plot manager
import os

out_dir=os.path.abspath('../../output/')

fig, ax=plt.subplots(figsize=(20, 10))
training_set.hist(ax=ax, bins=50, color='black', alpha=0.7)
plt.show()
fig.savefig(os.path.join(out_dir, "Project_Quality_Check_Before_Transformation_v2"), dpi=500)



# 4. Extract clusters using chr_num and chr_pos

from sklearn.cluster import KMeans

from vector_data_pre import perform_clustering

clustered_training_set, clustered_validation_set, clustered_test_set=perform_clustering(training_set, validation_set, test_set)


# 5. Perform feature engineering

from sklearn.preprocessing import StandardScaler # import transformer

from vector_data_pre import scale

scaled_training_set, scaled_validation_set, scaled_test_set=scale(clustered_training_set, clustered_validation_set, clustered_test_set)


# 6. Plot histogram of transformed training features and confirm quality

fig, ax=plt.subplots(figsize=(20, 20))
scaled_training_set.hist(ax=ax, bins=50, color='black', alpha=0.7)
plt.show()
fig.savefig(os.path.join(out_dir, "Project_Quality_Check_After_Transformation_v2"), dpi=500)


