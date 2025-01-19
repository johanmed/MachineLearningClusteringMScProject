#!/usr/bin/env python

"""
This script uses cluster assignments and their indices (in data_indices_clusters.csv) to establish a relationship between clusters and data (in project_dataset_with_desc_full_desc.csv)
Extract full description of traits
Compute statistics of each trait for each cluster

"""

# Read files

f1=open('../../../../data_indices_clusters.csv')
f1_read=f1.readlines()
f1.close()

#f2=open('../../../../project_dataset_all_traits_with_desc_full_desc.csv')
f2=open('../../../../chunks/chunk0.csv') # for demo
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


# Link index to trait description

trait_desc={}

for (index, line) in enumerate(f2_read):
    chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc = line.split(',')
    trait_desc[index]=full_desc
    
print('The indices and corresponding trait descriptions are: \n', trait_desc)

# Use relationship between index and trait category to get clusters and trait description instances

clusters_trait_desc={}

for key in clusters.keys():
        for val in clusters[key]:
            if key in clusters_trait_desc:
                clusters_trait_desc[key].append(trait_desc[val]) # append the trait category corresponding to val or index
            else:
                clusters_trait_desc[key]=[trait_desc[val]] # replace val or index by the trait category
                
print('The clusters and trait descriptions are: \n', clusters_trait_desc)


# Compute trait frequency in each cluster

clusters_trait_freq={}

for cluster in clusters_trait_categ.keys():
    traits=clusters_trait_desc[cluster]
    for trait in traits:
        if trait in clusters_trait_freq[cluster].keys(): # check if the trait category already in dictionary of the cluster
            clusters_trait_freq[cluster][trait] +=1 # add 1 to the count of the trait category if yes
        else:
            clusters_trait_freq[cluster]={trait:0} # initialize trait with count 0
    
print('The trait frequencies by cluster are :\n', clusters_trait_freq)  

# Generate causality networks
