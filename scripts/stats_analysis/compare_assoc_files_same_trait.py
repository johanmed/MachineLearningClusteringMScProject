#!/usr/bin/env python

"""
Read association info in csv file
Take rows concerning same trait
Zoom in chromosome number and marker position
Assuming they are identical, compare the p-values
"""

# Read association info

f=open('../../../../project_dataset_with_desc_full_desc.csv')
assoc_info=f.readlines()
f.close()

# Store each trait and association data in dictionary for efficient lookup

container={}

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
    
    

def analyze_traits(container):
    """
    Scrutinize and perform analysis of all the traits in container
    """
    assoc_diff={}
    
    for trait in container.keys():
        print(f'Analyzing trait {trait}...')
        trait_diff=compare_info_trait(container[trait])
    
        if len(trait_diff)>=2:
            print(f'The trait {trait} shows differences in p-lrt in at least 2 genomic positions that could be statistically meaningful')
    
            assoc_diff[trait]=trait_diff
    
    return assoc_diff
    

# Analyze each trait and search for differences in p_lrt that might be relevant statistically
    
results=analyze_traits(container)

#print('The raw results of the analysis are: \n', results)


# Plot pie chart of proportion of traits with differences in their association results across datasets

import matplotlib.pyplot as plt

fig, ax=plt.subplots()
ax.pie([len(results), len(container)-len(results)], labels=['Traits with differences', 'Traits with no differences'], rotatelabels=True)
ax.set_title('Proportion of traits with differences in association results')
ax.legend()

fig.savefig('../../output/proportion_traits_with_differences.png', dpi=500)


# Plot horizontal bar plot of proportion of differences for all traits with differences in association results

import numpy as np

fig, ax=plt.subplots()

ys=[i*10 for i in range(1, len(results)+1)] # place each at location determine by its index*10


y_labels=[j for j in results.keys()] # use trait names stored as keys

xs=np.linspace(0, 1, num=10, dtype=float)

widths=[len(results[j])/1 for j in results.keys()] # use the number of differences in each trait to determine the width of each bar

ax.barh(ys, widths, color='black')

ax.set_xticks(xs)

ax.set_yticks(ys)

ax.set_yticklabels(y_labels, rotation=30, fontsize=5)

ax.set_title('Proportion of differences in association results')

fig.savefig('../../output/proportion_differences_association_results.png', dpi=500)
