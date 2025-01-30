#!/usr/bin/env python

"""
Read association info in csv file
Take rows concerning same trait
Zoom in chromosome number and marker position
Assuming they are identical, compare the p-values
"""

# Module

def store_assoc_data(container, file):
    """
    Read association info
    Store each trait and association data in dictionary for efficient lookup
    """
    
    f=open(file) # for demo
    assoc_info=f.readlines()
    f.close()
    
    to_process=assoc_info[1:] # skip first line that is file header
    
    num_processing=len(to_process)
    
    for row in to_process:
        num_processing -= 1
        print(f'{num_processing} remaining lines to process')
        chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc=row.split(',')
        full_desc=full_desc.strip('\n')
        desc_full_desc = desc + ' ' + full_desc
        chr_num_pos= chr_num + ' ' + pos
        
        found_similar=False
        
        for key in container.keys():
            diff=set(desc_full_desc.split(' ')).difference(set(key.split(' ')))
            if 0<=len(diff)<=5: # similarity between two traits descriptions found if number of difference in words <= 5
                if chr_num_pos in container[key].keys():
                    container[key][chr_num_pos].append(float(p_lrt))
                else:
                    container[key][chr_num_pos]=[float(p_lrt)]
                        
                found_similar=True
                break
        
        if not found_similar:
            container[desc_full_desc] = {chr_num_pos: [float(p_lrt)]}
            
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
import json

container={}

# 1. Read data on disk for container if exists

if os.path.exists('../../../../container.json'):
    f1=open('../../../../container.json')
    json_content=f1.read()
    f1.close()
    
    json_content2dict=json.loads(json_content)
    #print(json_content2dict)
    
    container=json_content2dict

file='../../../../chunks/chunk0.csv'

#print(container)

dict_data=store_assoc_data(container, file)

#print(dict_data)

# 2. Save new dictionary (container) on disk

f2=open('../../../../container.json', 'w')

dict2json_content=json.dumps(dict_data)
f2.write(dict2json_content)

f2.close()




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


"""



# Main 4

# Plot vertical bar plot of proportion of differences for all traits with differences in association results

"""

fig, ax=plt.subplots(figsize=(20, 10))

data=pd.DataFrame([len(results[j]) for j in results.keys()], index=[j for j in results.keys()]) # use the number of differences in each trait to determine the width of each bar

data.plot.barh(ax=ax, color='black', alpha=0.7, rot=0.2, legend=False, position=1)

ax.set_title('Proportion of differences in association results')

fig.savefig('../../output/Proportion_differences_association_results.png', dpi=500)

"""



