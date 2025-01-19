#!/usr/bin/env python

"""
Summary:
This file contains code to load data from dataset for unsupervised machine learning
Data of each column are plotted in histogram to assess quality
Training, validation and test sets are defined from X data randomly
Features of the training set are scaled and validation and test set are transformed accordingly
Data of each column of the training are plotted in histogram to confirm quality
"""


# 1. Read data by chunks

import pandas as pd

#chunks= pd.read_csv('../../../../project_dataset_all_traits_with_desc_full_desc.csv', index_col=False, chunksize=1e7) # data read in chunks

"""
for ind, chunk in enumerate(chunks):
    chunk.to_csv(f'../../../../chunks/chunk{ind}.csv', index=False) # Save chunks on disk for ease of reading and referencing
"""

import os

"""
chunks_files=[os.path.join('../../../../chunks/', file) for file in os.listdir('../../../../chunks/')]

list_chunks=[] # store dataframes in a list

for file in chunks_files:
    list_chunks.append(pd.read_csv(file, index_col=False))

#print(list_chunks[0].head())

"""

sample_chunk=pd.read_csv('../../../../chunks/chunk0.csv', index_col=False) # for demo

# 2. Define training, validation and test sets

from sklearn.model_selection import train_test_split # import utility for splitting

def define_sets(X):
    processed_X=pd.DataFrame(X[['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt', 'desc']], dtype=float)
    #print('Processed_X looks like: \n', processed_X.head())

    X_train_valid, X_test= train_test_split(processed_X, test_size=0.1, random_state=2024)
    X_train, X_valid=train_test_split(X_train_valid, test_size=0.1, random_state=2024)

    #print('X_train looks like: \n', X_train.head())
    
    return X_train, X_valid, X_test

training_set, validation_set, test_set=define_sets(sample_chunk) # for demo

#training_sets, validation_sets, test_sets=[define_sets(X)[0] for X in list_chunks], [define_sets(X)[1] for X in list_chunks], [define_sets(X)[2] for X in list_chunks]

# 4. Plot histogram of training features and assess quality

import matplotlib.pyplot as plt # import plot manager

out_dir=os.path.abspath('../../output/')

"""
fig, axes=subplots(nrows=2, ncols=int(len(training_sets)/2), figsize=(30, 20))

for ind1, element in enumerate(training_sets):
    for ind2, ax in enumerate(axes):
        if ind1==ind2:, fig_size=(20, 10)
            element.hist(ax=ax, bins=50, color='black', alpha=0.2)
"""

training_set.hist(bins=50, color='black', alpha=0.2) # for demo
plt.show()
plt.savefig(os.path.join(out_dir, "Project_Quality_Check_Before_Transformation"), dpi=500)



# 5. Perform OneHotEncoding of trait category


from sklearn.preprocessing import OneHotEncoder
import numpy as np

def hot_encode(X_train, X_valid, X_test):
    desc_encoder=OneHotEncoder(sparse_output=False)
    
    desc_encoder1=desc_encoder.fit_transform((np.array(X_train['desc'])).reshape(-1, 1))
    
    X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4']]=desc_encoder1

    desc_encoder2=desc_encoder.transform((np.array(X_valid['desc'])).reshape(-1, 1))
    X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4']]=desc_encoder2

    desc_encoder3=desc_encoder.transform((np.array(X_test['desc'])).reshape(-1, 1))
    X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4']]=desc_encoder3

    return X_train, X_valid, X_test

encoded_training_set, encoded_validation_set, encoded_test_set=hot_encode(training_set, validation_set, test_set) # for demo

#encoded_training_sets=[hot_encode(a, b, c)[0] for a, b, c in zip(training_sets, validation_sets, test_sets)]
#encoded_validation_sets=[hot_encode(a, b, c)[1] for a, b, c in zip(training_sets, validation_sets, test_sets)]
#encoded_test_sets=[hot_encode(a, b, c)[2] for a, b, c in zip(training_sets, validation_sets, test_sets)]

# 6. Extract clusters using OneHotEncoding categories and p_lrt

from sklearn.cluster import KMeans

def perform_clustering_one(X_train, X_valid, X_test):
    prelim1_clustering=KMeans(n_clusters=5, algorithm='elkan', random_state=2024)

    prelim1_clustering1=prelim1_clustering.fit_transform(np.array(X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4', 'p_lrt']]).reshape(-1, 5))
    X_train[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering1

    prelim1_clustering2=prelim1_clustering.transform(np.array(X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4', 'p_lrt']]).reshape(-1, 5))
    X_valid[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering2

    prelim1_clustering3=prelim1_clustering.transform(np.array(X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'one_hot_desc4', 'p_lrt']]).reshape(-1, 5))
    X_test[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering3

    return X_train, X_valid, X_test
 
 
clustered1_training_set, clustered1_validation_set, clustered1_test_set=perform_clustering_one(encoded_training_set, encoded_validation_set, encoded_test_set) # for demo

#clustered1_training_sets=[perform_clustering_one(a, b, c)[0] for a, b, c in zip(encoded_training_sets, encoded_validation_sets, encoded_test_sets)]
#clustered1_validation_sets=[perform_clustering_one(a, b, c)[1] for a, b, c in zip(tencoded_training_sets, encoded_validation_sets, encoded_test_sets)]
#clustered1_test_sets=[perform_clustering_one(a, b, c)[2] for a, b, c in zip(encoded_training_sets, encoded_validation_sets, encoded_test_sets)]


# 7. Extract clusters using chr_num and chr_pos

def perform_clustering_two(X_train, X_valid, X_test):
    prelim2_clustering=KMeans(n_clusters=5, algorithm='elkan', random_state=2024)

    prelim2_clustering1=prelim2_clustering.fit_transform(np.array(X_train[['chr_num', 'pos']]).reshape(-1, 2))
    X_train[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering1

    prelim2_clustering2=prelim2_clustering.transform(np.array(X_valid[['chr_num', 'pos']]).reshape(-1, 2))
    X_valid[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering2

    prelim2_clustering3=prelim2_clustering.transform(np.array(X_test[['chr_num', 'pos']]).reshape(-1, 2))
    X_test[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering3

    return X_train, X_valid, X_test


clustered2_training_set, clustered2_validation_set, clustered2_test_set=perform_clustering_two(clustered1_training_set, clustered1_validation_set, clustered1_test_set) # for demo

#clustered2_training_sets=[perform_clustering_two(a, b, c)[0] for a, b, c in zip(clustered1_training_sets, clustered1_validation_sets, clustered1_test_sets)]
#clustered2_validation_sets=[perform_clustering_two(a, b, c)[1] for a, b, c in zip(clustered1_training_sets, clustered1_validation_sets, clustered1_test_sets)]
#clustered2_test_sets=[perform_clustering_two(a, b, c)[2] for a, b, c in zip(clustered1_training_sets, clustered1_validation_sets, clustered1_test_sets)]


# 8. Perform feature engineering

from sklearn.preprocessing import StandardScaler # import transformer

def scale(X_train, X_valid, X_test):
    std_scaler=StandardScaler()
    for i in X_train.columns:
        if i=='desc' or i=='full_desc':
            continue
        else:
            std_scaler1=std_scaler.fit_transform((np.array(X_train[i])).reshape(-1, 1)) # fit transformer on training set
            X_train['transformed_'+ i]=std_scaler1
    
            std_scaler2=std_scaler.transform((np.array(X_valid[i])).reshape(-1, 1)) # transform validation set
            X_valid['transformed_'+ i]=std_scaler2
    
            std_scaler3=std_scaler.transform((np.array(X_test[i])).reshape(-1, 1)) # transform test set
            X_test['transformed_'+ i]=std_scaler3

    return X_train, X_valid, X_test


scaled_training_set, scaled_validation_set, scaled_test_set=scale(clustered2_training_set, clustered2_validation_set, clustered2_test_set) # for demo

#scaled_training_sets=[scale(a, b, c)[0] for a, b, c in zip(clustered2_training_sets, clustered2_validation_sets, clustered2_test_sets)]
#scaled_validation_sets=[scale(a, b, c)[1] for a, b, c in zip(clustered2_training_sets, clustered2_validation_sets, clustered2_test_sets)]
#scaled_test_sets=[scale(a, b, c)[2] for a, b, c in zip(clustered2_training_sets, clustered2_validation_sets, clustered2_test_sets)]

# 9. Plot histogram of transformed training features and confirm quality

"""
fig, axes=subplots(nrows=2, ncols=int(len(scaled_training_sets)/2), figsize=(30, 20))

for ind1, element in enumerate(scaled_training_sets):
    for ind2, ax in enumerate(axes):
        if ind1==ind2:
            element.hist(ax=ax, bins=50, color='black', alpha=0.2)
"""
scaled_training_set.hist(bins=50, color='black', alpha=0.2)
plt.show()
plt.savefig(os.path.join(out_dir, "Project_Quality_Check_After_Transformation"), dpi=500)


# 10. Wrap up all transformations in a Transformer and add PCA to 2d for one_hot_desc, p_lrt, chr_num and pos

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

custom_preprocessing=Pipeline([('cluster', KMeans(n_clusters=5, algorithm='elkan', random_state=2024)), ('standardize', StandardScaler()), ('reduce', PCA(n_components=2, random_state=2024))])

preprocessing_hits=ColumnTransformer([('one_hot_desc_plrt_chr_num_pos', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos'])], remainder=StandardScaler())

preprocessing_qtl=ColumnTransformer([('one_hot_desc_plrt_chr_num', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num'])], remainder=StandardScaler())
