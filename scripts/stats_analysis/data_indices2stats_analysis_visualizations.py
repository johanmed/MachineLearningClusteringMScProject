#!/usr/bin/env python

"""
This script uses cluster assignments and their indices (in data_indices_clusters.csv) to establish a relationship between clusters and data (in project_dataset_all_traits_p_lrt_filtered.csv)
Extract trait categories
Compute statistics of each trait category for each cluster
"""

def sort_first_el(seq):
    return seq[0]
    
def sort_second_el(seq):
    return seq[1]

# 1, Read file and import training_validation_set

f=open('../../../../data_indices_clusters.csv')
f_read=f.readlines()
f.close()

from vector_data_post import training_validation_set

#print('index length', len(f_read))
#print('data length', training_validation_set.shape)

# 2. Extract clusters and elements

clusters={}

for line in f_read:
    ind, clust = line.split(',')
    clust=int(clust.strip('\n'))
    if clust in clusters.keys():
        clusters[clust].append(ind)
    else:
        clusters[clust]=[ind]
        
#print('The clusters and elements are: \n', clusters)

# 3. Link index to trait category and description

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

# 4. Use relationship between index and trait category to get clusters and trait category instances

clusters_trait_categ_desc={}

for cluster in clusters.keys():
    clusters_trait_categ_desc[cluster]=[]
    
    for val in clusters[cluster]:
        if val not in trait_categ_desc.keys():
            continue
        clusters_trait_categ_desc[cluster].append(trait_categ_desc[str(val)]) # append the trait category and description corresponding to val or index
        

#print('The clusters and corresponding category and description are: \n', clusters_trait_categ_desc)


# 5. Compute trait category and description frequencies in each cluster

clusters_trait_categ_freq={}
clusters_trait_desc_freq={}

for cluster in sorted(clusters_trait_categ_desc.keys()):
    
    interm_trait={}
    interm_desc={}
    
    traits=clusters_trait_categ_desc[cluster]
    
    for trait, desc in sorted(traits, key=sort_first_el):
    
        if trait in interm_trait.keys(): # check if the trait category already in dictionary of the cluster
            interm_trait[trait] += 1 # add 1 to the count of the trait category if yes
        else:
            interm_trait[trait] = 1 # initialize trait with count 1
            
    
        if desc in interm_desc.keys():
            interm_desc[desc] += 1 # add 1 to the count of the trait desc if yes
        else:
            interm_desc[desc] = 1 # initial desc with count 1
    
    clusters_trait_categ_freq[cluster] = interm_trait
    clusters_trait_desc_freq[cluster] = interm_desc


#print('The trait category frequencies by cluster are :\n', clusters_trait_categ_freq)  


# 6. Plot proportion of shared genetic features by trait categories in each cluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2024)

clusters_traits=pd.DataFrame(clusters_trait_categ_freq)

clusters_traits_transposed=clusters_traits.transpose()

fig, ax = plt.subplots(figsize=(20, 10))

clusters_traits_transposed.plot.bar(ax=ax)

ax.set_ylabel('Number of shared genetic features', fontsize=15)

ax.set_xlabel('Clusters', fontsize=15)

ax.set_title('Proportion of shared genetic features by trait categories in each cluster', fontsize=20)

ax.legend(['Diabetes trait', 'Immune system', 'Gastrointestinal trait', 'Undetermined'], loc='upper left')

plt.show()

fig.savefig('../../output/Association_traits_diabetes_vs_others.png', dpi=500)



# 7. Extract description of traits found to be associated, the number of shared genetic features and build collection of traits of interest

# Get trait category frequencies with mean and std for diabetes trait

trait_categ_freq={}
sample_diabetes_trait_freq=[]

for cluster in clusters_trait_categ_freq.keys():
    categ_freqs=clusters_trait_categ_freq[cluster]
    
    sample_diabetes_trait_freq.append(categ_freqs[0]) # add frequency of diabetes traits of the cluster to the sample
    
    for categ in categ_freqs.keys():
        if categ in trait_categ_freq.keys():
            trait_categ_freq[categ] += categ_freqs[categ]
        else:
            trait_categ_freq[categ] = categ_freqs[categ]
            
mean_df=np.mean(sample_diabetes_trait_freq) # compute mean of number of genetic features for diabetes traits over sample
std_df=np.std(sample_diabetes_trait_freq) # compute standard deviation of number of genetic features for diabetes traits over sample

# Get median of trait description frequencies

sample_trait_desc_freq=[]

for cluster in clusters_trait_desc_freq.keys():
    descs=clusters_trait_desc_freq[cluster]
    for desc in descs.keys():
        sample_trait_desc_freq.append(descs[desc])
        
mean_freq=np.mean(sample_trait_desc_freq)
std_freq=np.std(sample_trait_desc_freq)

#print('threshold set: ', mean_freq + std_freq)

assoc_desc_freq={}
diabetes_desc_freq={}
immune_desc_freq={}
gastro_desc_freq={}
undetermined_desc_freq={}

collection_traits_interest=[]

for cluster in clusters_trait_desc_freq.keys():

    descs=clusters_trait_desc_freq[cluster]
    categs=clusters_trait_categ_freq[cluster]
    vals=clusters_trait_categ_desc[cluster]
    
    container=[]
    
    if mean_df - std_df <= categs[0] <= mean_df + std_df: # discard cluster where number of genetic features for diabetes traits are beyond range estimated
        for desc in descs.keys():
            
            if ([1, desc] in vals or [2, desc] in vals or [3, desc] in vals) and descs[desc] > mean_freq + std_freq: # select only traits not related to diabetes
                
                
                # collect associated traits frequencies
                if desc in assoc_desc_freq.keys():
                    assoc_desc_freq[desc] += descs[desc] # update frequency according to descs[desc]
                else:
                    if type(desc) != str:
                        continue
                    else:
                        assoc_desc_freq[desc] = descs[desc] # set the frequency to descs[desc]
                
                container.append(desc) # store
                
                
                # collect frequencies of immune traits associated
                if [1, desc] in vals:
                    if desc in immune_desc_freq.keys():
                        immune_desc_freq[desc] += descs[desc]
                    else:
                        immune_desc_freq[desc] = descs[desc]
                
                
                # collect frequencies of gastro traits associated
                if [2, desc] in vals:
                    if desc in gastro_desc_freq.keys():
                        gastro_desc_freq[desc] += descs[desc]
                    else:
                        gastro_desc_freq[desc] = descs[desc]
                        
                
                # collect frequencies of undetermined traits associated
                if [3, desc] in vals:
                    if desc in undetermined_desc_freq.keys():
                        undetermined_desc_freq[desc] += descs[desc]
                    else:
                        undetermined_desc_freq[desc] = descs[desc]
                
            elif descs[desc] > mean_freq + std_freq: # select diabetes trait of relevance
                
                container.append(desc) # store                
                
                # collect frequencies of relevant diabetes traits
                if desc in diabetes_desc_freq.keys():
                    diabetes_desc_freq[desc] += descs[desc]
                else:
                    diabetes_desc_freq[desc] = descs[desc]
    
    
    collection_traits_interest.append(container)
                
#print('The number of traits found associated to diabetes traits are: ', len(assoc_desc_freq.keys()))

#print('The traits found associated to diabetes traits are: ', assoc_desc_freq)

#print('The collection of traits of interest for network analysis is :\n', collection_traits_interest)



# 8. Plot number of shared genetic features for selection of specific traits with diabetes traits

sorted_assoc_des=sorted(assoc_desc_freq.items(), key=sort_second_el, reverse=True)

top20_assoc_desc=[pair[0] for pair in sorted_assoc_des[1:20] if type(pair[0])==str and len(pair[0]) >= 5] # omit first element because it is nan

freq_top20_assoc_desc=[assoc_desc_freq[key] for key in top20_assoc_desc]

top20_assoc_desc = [' '.join(desc.split(' ')[:5]) for desc in top20_assoc_desc] # just keep the first 7 words of desc for labeling

#print('top20 looks like: ', top20_assoc_desc)

assoc=pd.DataFrame(freq_top20_assoc_desc, index=top20_assoc_desc)

fig, ax = plt.subplots(figsize=(20, 20))

assoc.plot.barh(ax=ax, legend=False, color='black', alpha=0.7)

ax.set_xlabel('Number of shared genetic features', fontsize=15)

ax.set_title('Proportion of genetic features shared by specific associated traits', fontsize=20)

plt.show()

fig.savefig('../../output/Proportion_genetic_features_shared_selection_specific_traits.png', dpi=500)



# 9. Compute probability that 2 specific traits are picked together


def compute_rel_freq(dic):
    new_dic={}
    total=sum(dic.values())
    for key in dic.keys():
        new_dic[key]=dic[key]/total
    return new_dic
    

trait_categ_freq = compute_rel_freq(trait_categ_freq)
assoc_desc_freq = compute_rel_freq(assoc_desc_freq)
diabetes_desc_freq = compute_rel_freq(diabetes_desc_freq)
immune_desc_freq = compute_rel_freq(immune_desc_freq)
gastro_desc_freq = compute_rel_freq(gastro_desc_freq)
undetermined_desc_freq = compute_rel_freq(undetermined_desc_freq)

intensities_2d=[]

for diabetes_trait in diabetes_desc_freq.keys():
    row=[]
    
    for assoc_trait in assoc_desc_freq.keys():
        
        if assoc_trait in immune_desc_freq.keys():
        
            row.append(diabetes_desc_freq[diabetes_trait] * immune_desc_freq[assoc_trait] * assoc_desc_freq[assoc_trait] * trait_categ_freq[1]) # compute probability based on product rule
            
        elif assoc_trait in gastro_desc_freq.keys():
            
            row.append(diabetes_desc_freq[diabetes_trait] * gastro_desc_freq[assoc_trait] * assoc_desc_freq[assoc_trait] * trait_categ_freq[2])
        
        elif assoc_trait in undetermined_desc_freq.keys():
            
            row.append(diabetes_desc_freq[diabetes_trait] * undetermined_desc_freq[assoc_trait] * assoc_desc_freq[assoc_trait] * trait_categ_freq[3])
    
    
    intensities_2d.append(row)
    
pd_intensities=pd.DataFrame(intensities_2d, columns=list(assoc_desc_freq.keys()), index=list(diabetes_desc_freq.keys()))

# 10. Plot intensity of trait to trait association

import seaborn as sns

plt.figure(figsize=(20, 20))

ax=sns.heatmap(pd_intensities, cbar=True)

ax.set_title('Strength of trait to trait association', fontsize=20)

plt.savefig('../../output/Association_intensity_trait2trait.png', dpi=500)
