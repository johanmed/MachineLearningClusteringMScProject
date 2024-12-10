#!/usr/bin/env python

"""
Summary:
This file contains code to load data from vector database and stores in variable X for unsupervised machine learning
X is converted to a dataframe
Data of each column are plotted in histogram to assess quality
Training, validation and test sets are defined from X data randomly
Features of the training set are scaled and validation and test set are transformed accordingly
Data of each column of the training are plotted in histogram to confirm quality
"""

# 1. Load data from vector database

import lmdb # import lmdb module to manage access to the vector database
from struct import * # manage unpacking operations

X=[] # empty array to keep data

import os

database=os.path.abspath('/home/johannes/Downloads/real_project.mdb')

with lmdb.open(database, subdir=False) as env:
    with env.begin() as txn:
        with txn.cursor() as curs:
            for (key, value) in list(txn.cursor().iternext()):
                if key==b'meta':
                    continue
                else:
                    chr, pos, se, l_mle, p_lrt= unpack('>cLfff', key)
                    chr_num=int.from_bytes(chr)
                    af, beta, se, l_mle, p_lrt, desc= unpack('=ffffff', value)
                    X.append([chr_num, pos, af, beta, se, l_mle, p_lrt, desc])

print('The size of the collection is: ', len(X)) # check the size of X

# 2. Convert data into dataframe

import pandas as pd
import numpy as np
new_X= pd.DataFrame(np.array(X), columns=['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt', 'desc'])
#new_X.to_csv('../data/whole_dataset_desc_full_desc.csv', index=False)


# 3. Define training, validation and test sets

from sklearn.model_selection import train_test_split # import utility for splitting

X_train_valid, X_test= train_test_split(new_X, test_size=0.1, random_state=2024)
X_train, X_valid=train_test_split(X_train_valid, test_size=0.1, random_state=2024)


# 4. Plot histogram of training features and assess quality

import matplotlib.pyplot as plt # import plot manager

X_train.hist(bins=50, figsize=(10, 10))
out_dir=os.path.abspath('../output/')
plt.savefig(os.path.join(out_dir, "Project Quality check before transformation"))



# 5. Perform OneHotEncoding of trait category


from sklearn.preprocessing import OneHotEncoder

desc_encoder=OneHotEncoder(sparse_output=False)

desc_encoder1=desc_encoder.fit_transform((np.array(X_train['desc'])).reshape(-1, 1))
X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3']]=desc_encoder1

desc_encoder2=desc_encoder.transform((np.array(X_valid['desc'])).reshape(-1, 1))
X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3']]=desc_encoder2

desc_encoder3=desc_encoder.transform((np.array(X_test['desc'])).reshape(-1, 1))
X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3']]=desc_encoder3


# 6. Extract clusters using OneHotEncoding categories and p_lrt

from sklearn.cluster import KMeans

prelim1_clustering=KMeans(n_clusters=5, algorithm='elkan', random_state=2024)

prelim1_clustering1=prelim1_clustering.fit_transform(np.array(X_train[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt']]).reshape(-1, 4))
X_train[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering1

prelim1_clustering2=prelim1_clustering.transform(np.array(X_valid[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt']]).reshape(-1, 4))
X_valid[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering2

prelim1_clustering3=prelim1_clustering.transform(np.array(X_test[['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt']]).reshape(-1, 4))
X_test[['combined_desc_p_lrt1', 'combined_desc_p_lrt2', 'combined_desc_p_lrt3', 'combined_desc_p_lrt4', 'combined_desc_p_lrt5']]=prelim1_clustering3


# 7. Extract clusters using chr_num and chr_pos

from sklearn.cluster import KMeans

prelim2_clustering=KMeans(n_clusters=5, algorithm='elkan', random_state=2024)

prelim2_clustering1=prelim2_clustering.fit_transform(np.array(X_train[['chr_num', 'pos']]).reshape(-1, 2))
X_train[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering1

prelim2_clustering2=prelim2_clustering.transform(np.array(X_valid[['chr_num', 'pos']]).reshape(-1, 2))
X_valid[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering2

prelim2_clustering3=prelim2_clustering.transform(np.array(X_test[['chr_num', 'pos']]).reshape(-1, 2))
X_test[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim2_clustering3




# 8. Perform feature engineering

from sklearn.preprocessing import StandardScaler # import transformer

std_scaler=StandardScaler()
for i in X_train.columns:
    if i=='desc':
        continue
    else:
        std_scaler1=std_scaler.fit_transform((np.array(X_train[i])).reshape(-1, 1)) # fit transformer on training set
        X_train['transformed_'+ i]=std_scaler1
    
        std_scaler2=std_scaler.transform((np.array(X_valid[i])).reshape(-1, 1)) # transform validation set
        X_valid['transformed_'+ i]=std_scaler2
    
        std_scaler3=std_scaler.transform((np.array(X_test[i])).reshape(-1, 1)) # transform test set
        X_test['transformed_'+ i]=std_scaler3



# 9. Plot histogram of transformed training features and confirm quality

X_train.hist(bins=50, figsize=(25, 25))
plt.savefig(os.path.join(out_dir, "Project Quality check after transformation"))


# 10. Wrap up all transformations in a Transformer and add PCA to 2d for one_hot_desc, p_lrt, chr_num and pos

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

custom_preprocessing=Pipeline([('cluster', KMeans(n_clusters=5, algorithm='elkan', random_state=2024)), ('standardize', StandardScaler()), ('reduce', PCA(n_components=2, random_state=2024))])

preprocessing_hits=ColumnTransformer([('one_hot_desc_plrt_chr_num_pos', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos'])], remainder=StandardScaler())

preprocessing_qtl=ColumnTransformer([('one_hot_desc_plrt_chr_num', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num'])], remainder=StandardScaler())
