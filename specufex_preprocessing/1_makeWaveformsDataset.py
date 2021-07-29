#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:19 2021
example: Parkfield repeaters::
@author: theresasawi
"""

import argparse
import glob
import os

import h5py
import numpy as np
import obspy
import pandas as pd
import yaml

from functions.generators import gen_wf_from_folder

parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

#%% load config file

with open(args.config_filename, 'r') as stream:
    try:
        # print(yaml.safe_load(stream))

        config = yaml.safe_load(stream)

    except yaml.YAMLError as exc:
        print(exc)

#%% Go through levels of config file (headers), and save all values
## to the same list. Then, assign those values to standard variables


variable_list = []
value_list = []


for headers in config.values():

    for value_dict in headers:
        print(value_dict)

        for k, v in value_dict.items():
            print(v,k)

            variable_list.append(k)
            value_list.append(v)



## assign values to variables
for index, value in enumerate(value_list):
    exec(f"{variable_list[index]} = value")



#%%






#%%


# ==============================================
# STUFF THAT GETS CHANGED WHEN WE MOVE TO config.py
#%% load project variables: names and paths


# key = sys.argv[1]
# key = 'Parkfield_Repeaters'
print(key)


dataH5_name = f'data_{key}.hdf5'

dataH5_path = projectPath + '/H5files/' + dataH5_name

wf_cat_out = projectPath + 'wf_cat_out.csv'


if not os.path.isdir(projectPath + '/H5files/'):
    os.mkdir(projectPath + '/H5files/')






#%% get global catalog

cat = pd.read_csv(pathCat, header=None,delim_whitespace=True)


cat.columns = cat_columns


#%% get list of waveforms and sort

wf_filelist = glob.glob(pathWF + '*')
wf_filelist.sort()

wf_filelist = wf_filelist

wf_test = obspy.read(wf_filelist[0])

lenData = len(wf_test[0].data)



#%% define generator (function)


gen_wf = gen_wf_from_folder(wf_filelist,key,lenData,channel_ID)


## clear old H5 if it exists, or else error will appear
if os.path.exists(dataH5_path):
    os.remove(dataH5_path)

#%% add catalog and waveforms to H5


evID_keep = [] #list of wfs to keep

with h5py.File(dataH5_path,'a') as h5file:



    global_catalog_group =  h5file.create_group("catalog/global_catalog")


    for col in cat.columns:

        if col == 'datetime': ## if there are other columns in your catalog
        #that are stings, then you may need to extend conditional statement
        # to use the dtype='S' flag in the next line
            global_catalog_group.create_dataset(name='datetime',data=np.array(cat['datetime'],dtype='S'))

        else:
            exec(f"global_catalog_group.create_dataset(name='{col}',data=cat.{col})")


    waveforms_group  = h5file.create_group("waveforms")
    station_group = h5file.create_group(f"waveforms/{station}")
    channel_group = h5file.create_group(f"waveforms/{station}/{channel}")



    dupl_evID = 0 #duplicate event IDs?? not here, sister
    n=0

    while n <= len(wf_filelist): ## not sure a better way to execute this? But it works

        try:   #catch generator "stop iteration" error


            #these all defined in generator at top of script
            data, evID, n = next(gen_wf)
            if n%500==0:
                print(n, '/', len(wf_filelist))
            # if evID not in group, add dataset to wf group
            if evID not in channel_group:
                channel_group.create_dataset(name= evID, data=data)
                evID_keep.append(int(evID))
            elif evID in channel_group:
                dupl_evID += 1

        except StopIteration: #handle generator error
            break


    sampling_rate = wf_test[0].stats.sampling_rate
    # instr_response = wf_test[0].stats.instrument_response
    station_info = f"{wf_test[0].stats.network}.{wf_test[0].stats.station}.{wf_test[0].stats.location}.{wf_test[0].stats.channel}."
    calib = wf_test[0].stats.calib
    _format = wf_test[0].stats._format


    processing_group = h5file.create_group(f"{station}/processing_info")


    processing_group.create_dataset(name= "sampling_rate_Hz", data=sampling_rate)#,dtype='S')
    processing_group.create_dataset(name= "station_info", data=station_info)
    processing_group.create_dataset(name= "calibration", data=calib)#,dtype='S')
    processing_group.create_dataset(name= "orig_formata", data=_format)#,dtype='S')
    # processing_group.create_dataset(name= "instr_response", data=instr_response,dtype='S')
    processing_group.create_dataset(name= "lenData", data=lenData)#,dtype='S')






print(dupl_evID, ' duplicate events found and avoided')
print(n- dupl_evID, ' waveforms loaded')



#%% save final working catalog to csv


cat_keep_wf = cat[cat['event_ID'].isin(evID_keep)]

if os.path.exists(wf_cat_out):
    os.remove(wf_cat_out)


print(len(cat_keep_wf), ' events in wf catalog')


#%%
