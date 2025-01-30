#!/usr/bin/env python

"""
Read association info of 1 csv file (chunk) at a time
Take rows concerning same trait
Store data in dictionary to prevent duplication
"""


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
        
        if desc_full_desc in container.keys():
            if chr_num_pos in container[desc_full_desc].keys():
                container[desc_full_desc][chr_num_pos].append(float(p_lrt))
            else:
                container[desc_full_desc][chr_num_pos]=[float(p_lrt)]
                
        else:
            container[desc_full_desc] = {chr_num_pos: [float(p_lrt)]}
            
        #print('The container is: \n', container)
        #print('The length of the container is: ', len(container))
    
    return container



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

file='../../../../chunks/chunk12.csv' # all the chunks processed till the last (12th chunk)

dict_data=store_assoc_data(container, file)

print('Number of traits in dict_data is: ', len(dict_data.keys()))

# 2. Save new dictionary (container) on disk

f2=open('../../../../container.json', 'w')

dict2json_content=json.dumps(dict_data)
f2.write(dict2json_content)

f2.close()



