#!/usr/bin/env python

"""
Read association info in csv file
Take rows concerning same trait
Zoom in chromosome number and marker position
Assuming they are identical, compare the p-values
"""

# Read association info

f=open('../../../../project_dataset_all_traits_p_lrt_filtered.csv')
assoc_info=f.readlines()
f.close()

# Store each trait and association data in dictionary for efficient lookup
# Keys are (desc, full_desc) and values are dictionary where keys are (chr_num, pos) and arrays of p_lrt are values

container={}

for row in assoc_info[1:]: # skip first line that is file header
    chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc=row.split(',')
    full_desc=full_desc.strip('\n')
    if full_desc in container.keys():
        if (chr_num, pos) in (container[full_desc]).keys():
            container[(desc, full_desc)][(chr_num, pos)].append(float(p_lrt))
        else:
            container[(desc, full_desc)][(chr_num, pos)]=[float(p_lrt)]
    else:
        container[(desc, full_desc)]={(chr_num, pos):[float(p_lrt)]}
        
print('The container is: \n', container)
print('The length of the container is: ', len(container))

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
    

# Analyze each trait and search for differences in p_lrt that might be relevant statistically
    
results=analyze_traits(container, compare_info_trait)

#print('The raw results of the analysis are: \n', results)
print('The length of results: ', len(results))

# Plot pie chart of proportion of traits with differences in their association results across datasets

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
import numpy as np
from collections import Counter

fig, (ax1, ax2)=plt.subplots(1, 2)

# pie chart parameters
data=[len(results), len(container)-len(results)]
angle=-180 * data[0]
labels=['Traits with differences', 'Traits with no differences']
explode=(0.1, 0)

wedges, *so= ax1.pie(data, labels=labels, colors=['r', 'g'], autopct='%1.2f%%', explode=explode, labeldistance=1.0, startangle=angle)

# bar chart parameters
return_desc=[desc for (desc, full_desc) in results.keys()]
dic_desc= Counter(return_desc)
trait_categs=dic_desc.keys()
trait_categ_metadata={0:'Diabetes', 1: 'Immune system', 2: 'Gastrointestinal system', 3: 'Unknown'}
trait_categs=[trait_categ_metadata[key] for key in trait_categs]
freqs=dic_desc.values()
freqs=[val/sum(freqs) for val in freqs]
bottom=1
width=0.2

for j, (height, label) in enumerate([*zip(freqs, trait_categs)]):
    bottom -= height
    bar_container=ax2.bar(0, height, width, bottom=bottom, color='C0', label=label, alpha=1+0.25*j)
    ax2.bar_label(bar_container, labels=[f'{height:.0%}'], label_type='center')
    
ax2.set_title('Trait categories')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(-2.5 * width, 2.5 * width)

# Use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(freqs)

# Draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(4)
ax2.add_artist(con)

# Draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(4)

fig.suptitle('Proportion of traits with differences in association results')

plt.show()

fig.savefig('../../output/Proportion_traits_with_differences.png', dpi=500)


# Plot vertical bar plot of proportion of differences for all traits with differences in association results

fig, ax=plt.subplots(figsize=(20, 10))

data=pd.DataFrame([len(results[j]) for j in results.keys()], index=[j for j in results.keys()]) # use the number of differences in each trait to determine the width of each bar

data.plot.barh(ax=ax, color='black', alpha=0.7, rot=0.2, legend=False, position=1)

ax.set_title('Proportion of differences in association results')

fig.savefig('../../output/Proportion_differences_association_results.png', dpi=500)
