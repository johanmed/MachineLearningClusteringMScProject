#!/usr/bin/env python

"""
Summary:
This file contains code to load data from dataset for unsupervised machine learning
Data of each column are plotted in histogram to assess quality
Training, validation and test sets are defined from X data randomly
Features of the training set are scaled and validation and test set are transformed accordingly
Data of each column of the training are plotted in histogram to confirm quality
"""


# 1. Import data as dataframe

import pandas as pd

new_X= pd.read_csv('../../../../project_dataset_with_desc_full_desc.csv', index_col=False)

#print('new X looks like: \n', new_X.head())


# 2. Define training, validation and test sets

from sklearn.model_selection import train_test_split # import utility for splitting

processed_X=pd.DataFrame(new_X[['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt', 'desc']], dtype=float)

#print('processed_X is : \n', processed_X.head())

X_train_valid, X_test= train_test_split(processed_X, test_size=0.1, random_state=2024)

X_train, X_valid=train_test_split(X_train_valid, test_size=0.1, random_state=2024)

#print('X_train is: \n', X_train.head())

# 4. Plot histogram of training features and assess quality

import matplotlib.pyplot as plt # import plot manager

#X_train.hist(bins=50, figsize=(10, 10))
out_dir=os.path.abspath('../../output/')
#plt.savefig(os.path.join(out_dir, "Project_Quality_Check_Before_Transformation"))



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
    if i=='desc' or i=='full_desc':
        continue
    else:
        std_scaler1=std_scaler.fit_transform((np.array(X_train[i])).reshape(-1, 1)) # fit transformer on training set
        X_train['transformed_'+ i]=std_scaler1
    
        std_scaler2=std_scaler.transform((np.array(X_valid[i])).reshape(-1, 1)) # transform validation set
        X_valid['transformed_'+ i]=std_scaler2
    
        std_scaler3=std_scaler.transform((np.array(X_test[i])).reshape(-1, 1)) # transform test set
        X_test['transformed_'+ i]=std_scaler3



# 9. Plot histogram of transformed training features and confirm quality

#X_train.hist(bins=50, figsize=(25, 25))
#plt.savefig(os.path.join(out_dir, "Project_Quality_Check_After_Transformation"))


# 10. Wrap up all transformations in a Transformer and add PCA to 2d for one_hot_desc, p_lrt, chr_num and pos

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

custom_preprocessing=Pipeline([('cluster', KMeans(n_clusters=5, algorithm='elkan', random_state=2024)), ('standardize', StandardScaler()), ('reduce', PCA(n_components=2, random_state=2024))])

preprocessing_hits=ColumnTransformer([('one_hot_desc_plrt_chr_num_pos', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num', 'pos'])], remainder=StandardScaler())

preprocessing_qtl=ColumnTransformer([('one_hot_desc_plrt_chr_num', custom_preprocessing, ['one_hot_desc1', 'one_hot_desc2', 'one_hot_desc3', 'p_lrt', 'chr_num'])], remainder=StandardScaler())
