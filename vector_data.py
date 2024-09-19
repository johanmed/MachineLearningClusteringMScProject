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

database='gemma.mdb'

with lmdb.open(database, subdir=False) as env:
    with env.begin() as txn:
        with txn.cursor() as curs:
            for (key, value) in list(txn.cursor().iternext()):
                if key==b'meta':
                    continue
                else:
                    chr, pos= unpack('>cL', key)
                    chr_num=int.from_bytes(chr)
                    af, beta, se, l_mle, p_lrt= unpack('=fffff', value)
                    X.append([chr_num, pos, af, beta, se, l_mle, p_lrt])

print('The size of the collection is: ', len(X)) # check the size of X

# 2. Convert data into dataframe
import pandas as pd
import numpy as np
new_X= pd.DataFrame(np.array(X), columns=['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt'])


# 3. Define training, validation and test sets
from sklearn.model_selection import train_test_split # import utility for splitting
X_train_valid, X_test= train_test_split(new_X, test_size=0.1, random_state=2024)
X_train, X_valid=train_test_split(X_train_valid, test_size=0.1, random_state=2024)


# 4. Plot histogram of training features and assess quality

#import matplotlib.pyplot as plt # import plot manager
#X_train.hist(bins=50, figsize=(10, 10))
#plt.savefig("Quality check before transformation")
#plt.show()


# 5. Perform feature scaling
from sklearn.preprocessing import StandardScaler # import transformer

std_scaler=StandardScaler()
for i in ['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt']:
    std_scaler1=std_scaler.fit_transform((np.array(X_train[i])).reshape(-1, 1)) # fit transformer on training set
    X_train['transformed_'+ i]=std_scaler1
    del X_train[i]
    std_scaler2=std_scaler.transform((np.array(X_valid[i])).reshape(-1, 1)) # transform validation set
    X_valid['transformed_'+ i]=std_scaler2
    del X_valid[i]
    std_scaler3=std_scaler.transform((np.array(X_test[i])).reshape(-1, 1)) # transform test set
    X_test['transformed_'+ i]=std_scaler3
    del X_test[i]


# 6. Plot histogram of transformed training features and confirm quality

#X_train.hist(bins=50, figsize=(10, 10))
#plt.savefig("Quality check after transformation")
#plt.show()
