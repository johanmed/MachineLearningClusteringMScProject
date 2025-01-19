#!/usr/bin/env python

"""
This script uses cluster assignments and their indices (in data_indices_clusters.csv) to establish a relationship between clusters and data (in project_dataset_with_desc_full_desc.csv)
Extract trait categories
Compute statistics of each trait category for each cluster
Show proportions of immune and gastrointestinal system traits vs proportion of diabetes traits by cluster
"""

# Main 1

# Read files

f1=open('../../../../data_indices/data_indices_clusters0.csv') # need to change according to the chunk number
f1_read=f1.readlines()
f1.close()

f2=open('../../../../chunks/chunk0.csv') # need to change according to the chunk number
f2_read=f2.readlines()
f2.close()

# Extract clusters and elements

clusters={}

for line in f1_read:
    ind, clust = line.split(',')
    if clust in clusters.keys():
        clusters[clust].append(ind)
    else:
        clusters[clust]=[ind]
        
print('The clusters and elements are: \n', clusters)


# Link index to trait category

trait_categ={}

for (index, line) in enumerate(f2_read):
    chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc = line.split(',')
    trait_categ[index]=desc
    
print('The indices and corresponding trait categories are: \n', trait_categ)

# Use relationship between index and trait category to get clusters and trait category instances

import os

clusters_trait_categ={}

if os.path.exists('../../../../clusters_trait_categ.csv'):
    f3=open('../../../../clusters_trait_categ.csv')
    f3_read=f3.readlines()
    f3_read.close()
    for element in f3_read:
        cluster, categ = element.split(',')
        clusters_trait_categ[cluster]=categ

for key in clusters.keys():
        for val in clusters[key]:
            if key in clusters_trait_categ:
                clusters_trait_categ[key].append(trait_categ[val]) # append the trait category corresponding to val or index
            else:
                clusters_trait_categ[key]=[trait_categ[val]] # replace val or index by the trait category
                
print('The clusters and trait category instances are: \n', clusters_trait_categ)

# Store clusters and trait category instances on disk

f4=open('../../../../clusters_trait_categ.csv', 'w')
for key in clusters_trait_categ:
    f4.write(f'{key}, {clusters_trait_categ[key]}\n')







# Main 2

# Compute trait category frequency in each cluster

"""
clusters_trait_freq={}

for cluster in clusters_trait_categ.keys():
    traits=clusters_trait_categ[cluster]
    for trait in traits:
        if trait in clusters_trait_freq[cluster].keys(): # check if the trait category already in dictionary of the cluster
            clusters_trait_freq[cluster][trait] +=1 # add 1 to the count of the trait category if yes
        else:
            clusters_trait_freq[cluster]={trait:0} # initialize trait with count 0
    
print('The trait category frequencies by cluster are :\n', clusters_trait_freq)  

"""





# Main 3

# Plot proportion of each trait category by cluster

"""
import pandas as pd

clusters_traits=pd.DataFrame(clusters_trait_freq)
clusters_traits_transposed=clusters_traits.transpose()

import matplotlib.pyplot as plt

clusters_traits_transposed.plot.bar()

plt.savefig('../../output/traits_immune_gastro_vs_traits_diabetes.png', dpi=500)
"""
