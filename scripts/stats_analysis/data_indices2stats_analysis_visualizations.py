#!/usr/bin/env python

"""
This script uses cluster assignments and their indices (in data_indices_clusters.csv) to establish a relationship between clusters and data (in project_dataset_all_traits_p_lrt_filtered.csv)
Extract trait categories
Compute statistics of each trait category for each cluster
"""

# Read file and import training_validation_set

f=open('../../../../data_indices_clusters.csv')
f_read=f.readlines()
f.close()

from vector_data import training_validation_set

#print('index length', len(f_read))
#print('data length', training_validation_set.shape)

# Extract clusters and elements

clusters={}

for line in f_read:
    ind, clust = line.split(',')
    clust=int(clust.strip('\n'))
    if clust in clusters.keys():
        clusters[clust].append(ind)
    else:
        clusters[clust]=[ind]
        
#print('The clusters and elements are: \n', clusters)

# Link index to trait category and description

import numpy as np
import json
import os

if os.path.exists('../../../../trait_categ_desc.json'):

    file1=open('../../../../trait_categ_desc.json')
    json2dict=file1.read()
    file1.close()
    
    trait_categ_desc=json.loads(json2dict)
    
else:

    trait_categ_desc={}
    seen=[]

    tv_data=np.array(training_validation_set)
    #print('tv_data looks like:\n', tv_data[:10])

    len_processing=len(tv_data)
    #print('length', len_processing)

    for (index, line) in enumerate(tv_data): # skip header line
        len_processing -= 1
        print(f'{len_processing} more to process')
        chr_num, pos, desc, full_desc = line[0], line[1], line[3], line[4] # chr_num, pos, desc, full_desc are at index 0, 1, 3, 4 respectively
        if [chr_num, pos, full_desc] not in seen: # strategy to take care of duplicates
            trait_categ_desc[str(index)]=[int(desc), full_desc] # add only [desc, full_desc] not seen before
            seen.append([chr_num, pos, full_desc]) # update seen
    
    file2=open('../../../../trait_categ_desc.json', 'w')
    dict2json=json.dumps(trait_categ_desc)
    file2.write(dict2json)
    file2.close()


#print('trait_categ_desc', trait_categ_desc)

    
#print('The indices and corresponding trait categories are: \n', trait_categ_desc)

# Use relationship between index and trait category to get clusters and trait category instances

clusters_trait_categ_desc={}

for cluster in clusters.keys():
    clusters_trait_categ_desc[cluster]=[]
    
    for val in clusters[cluster]:
        if val not in trait_categ_desc.keys():
            continue
        clusters_trait_categ_desc[cluster].append(trait_categ_desc[str(val)]) # append the trait category and description corresponding to val or index
        
        
#print('The clusters and corresponding category and description are: \n', clusters_trait_categ_desc)

# Compute trait category and description frequencies in each cluster

clusters_trait_categ_freq={}
clusters_trait_desc_freq={}

for cluster in clusters_trait_categ_desc.keys():
    clusters_trait_categ_freq[cluster]={}
    clusters_trait_desc_freq[cluster]={}
    
    traits=clusters_trait_categ_desc[cluster]
    
    for trait, desc in traits:
    
        if trait in clusters_trait_categ_freq[cluster].keys(): # check if the trait category already in dictionary of the cluster
            clusters_trait_categ_freq[cluster][trait] += 1 # add 1 to the count of the trait category if yes
        else:
            clusters_trait_categ_freq[cluster]={trait:1} # initialize trait with count 1
            
    
        if desc in clusters_trait_desc_freq[cluster].keys():
            clusters_trait_desc_freq[cluster][desc] += 1 # add 1 to the count of the trait desc if yes
        else:
            clusters_trait_desc_freq[cluster]={desc:1} # initial desc with count 1
            
print('The trait category frequencies by cluster are :\n', clusters_trait_categ_freq)  


# Plot proportion of shared genetic features by trait categories in each cluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

clusters_traits=pd.DataFrame(clusters_trait_categ_freq)
clusters_traits_transposed=clusters_traits.transpose()

fig, ax = plt.subplots(figsize=(20, 10))

clusters_traits_transposed.plot.bar(ax=ax)

ax.set_ylabel('Number of shared genetic features')

ax.set_title('Proportion of shared genetic features by trait categories in each cluster')

plt.show()

fig.savefig('../../output/Association_traits_diabetes_vs_others.png', dpi=500)


# Extract description of traits found to be associated, the number of shared genetic features and build collection of traits of interest

sample_diabetes_trait_freq=[]

for cluster in clusters_trait_categ_freq.keys():
    categ_freqs=clusters_trait_categ_freq[cluster]
    sample_diabetes_trait_freq.append(categ_freqs[0]) # add frequency of diabetes traits of the cluster to the sample
    
mean_df=np.mean(sample_diabetes_trait_freq) # compute mean of number of genetic features for diabetes traits over sample
std_df=np.std(sample_diabetes_trait_freq) # compute standard deviation of number of genetic features for diabetes traits over sample

assoc_desc_freq={}

collection_traits_interest=[]

for cluster in clusters_trait_desc_freq.keys():
    descs=clusters_trait_desc_freq[cluster]
    categs=clusters_trait_categ_freq[cluster]
    if mean_df - std_df <= categs[0] <= mean_df + std_df: # discard cluster where number of genetic features for diabetes traits are beyond range estimated
        for desc in descs.keys():
            if [1, desc] in trait_categ_desc.values() or [2, desc] in trait_categ_desc.values() or [3, desc] in trait_categ_desc.values(): # select only traits not related to diabetes
                if desc in assoc_desc_freq.keys():
                    assoc_desc_freq[desc] += 1
                else:
                    assoc_desc_freq[desc]= 1
                    
                collection_traits_interest.append(desc) # add traits non related to diabetes to collection
            else:
                collection_traits_interest.append(desc) # add diabetes traits to collection
                
print('The number of traits found associated to diabetes traits are: ', len(desc_freq.keys()))

print('The collection of traits of interest for network analysis is :\n', collection_traits_interest)

# Plot number of shared genetic features for specific traits with diabetes traits

assoc=pd.DataFrame(assoc_desc_freq)

fig, ax = plt.subplots(figsize=(20, 10))

assoc.plot.barh(ax=ax, legend=False, color='black', alpha=0.7)

ax.set_xlabel('Number of shared genetic features')

ax.set_title('Proportion of genetic features shared by specific associated traits')

plt.show()

fig.savefig('../../output/Proportion_genetic_features_shared_specific_traits.png', dpi=500)

