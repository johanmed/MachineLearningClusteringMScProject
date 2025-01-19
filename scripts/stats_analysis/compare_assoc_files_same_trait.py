#!/usr/bin/env python

"""
Read association info in csv file
Take rows concerning same trait
Zoom in chromosome number and marker position
Assuming they are identical, compare the p-values
"""


def store_assoc_data(container, file):
    """
    Read association info
    Store each trait and association data in dictionary for efficient lookup
    """
    
    f=open(file) # for demo
    assoc_info=f.readlines()
    f.close()

    for row in assoc_info[1:]: # skip first line that is file header
        chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc=row.split(',')
        if full_desc in container.keys():
            if (chr_num, pos) in (container[full_desc]).keys():
                container[full_desc][(chr_num, pos)].append(float(p_lrt))
            else:
                container[full_desc][(chr_num, pos)]=[float(p_lrt)]
        else:
            container[full_desc]={(chr_num, pos):[float(p_lrt)]}
        
    #print('The container is: \n', container)
    #print('The length of the container is: ', len(container))
    
    return container


def compare_info_trait(trait_pos):
    """
    Compare p_lrt for each combination of chr_num and pos
    """
    
    diff=[]
    
    for loc in trait_pos.keys():
        mini=min(trait_pos[loc])
        maxi=max(trait_pos[loc])
        if (maxi-mini)>0.5: # cutoff to determine divergence in p_lrt set to 0.5
            diff.append([loc, [mini, maxi]])
    
    return diff
    
    

def analyze_traits(container, compare_info_trait):
    """
    Scrutinize and perform analysis of all the traits in container
    """
    assoc_diff={}
    
    for trait in container.keys():
        print(f'Analyzing trait {trait}...')
        trait_diff=compare_info_trait(container[trait])
    
        if len(trait_diff)>=1:
            print(f'The trait {trait} shows differences in p-lrt in at least 1 genomic position that could be statistically meaningful')
    
            assoc_diff[trait]=trait_diff
    
    return assoc_diff
    




# Main 1

import os

container={}

# 1. Read data on disk for container if exists

if os.path.exists('../../../../container.csv'):
    f1=open('../../../../container.csv')
    read=f.readlines()
    f1.close()
    for el in read:
        key, value=el.split(',')
        container[key]=value

file='../../../../chunks/chunk0.csv'

dict_data=store_assoc_data(container, file)

# 2. Save new dictionary (container) on disk

f2=open('../../../../container.csv', 'w')
for key in dict_data:
    f2.write(f'{key}, {dict_data[key]}\n')




# Main 2

# Proceed to actual analysis of each trait and search for differences in p_lrt that might be relevant statistically

"""
results=analyze_traits(dict_data, compare_info_trait)

#print('The raw results of the analysis are: \n', results)
#print('The length of results: ', len(results))
"""



# Main 3

# Plot pie chart of proportion of traits with differences in their association results across datasets after all data
"""
import matplotlib.pyplot as plt
import pandas as pd

fig, ax=plt.subplots()

data=[len(results), len(container)-len(results)]

labels=['Traits with differences', 'Traits with no differences']

explode=(0.4, 0) # explode the wedge of traits with differences by 0.4

ax.pie(data, labels=labels, colors=['r', 'g'], autopct='%1.2f%%')

ax.set_title('Proportion of traits with differences in association results')

fig.savefig('../../output/proportion_traits_with_differences.png', dpi=500)
"""



# Main 4

# Plot vertical bar plot of proportion of differences for all traits with differences in association results

"""
fig, ax=plt.subplots(figsize=(20, 10))

data=pd.DataFrame([len(results[j]) for j in results.keys()], index=[j for j in results.keys()]) # use the number of differences in each trait to determine the width of each bar

data.plot.barh(ax=ax, color='black', alpha=0.7, rot=0.2)

ax.set_title('Proportion of differences in association results')

fig.savefig('../../output/proportion_differences_association_results.png', dpi=500)
"""
