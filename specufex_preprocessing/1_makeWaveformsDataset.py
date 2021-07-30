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

from functions.generators import gen_wf_from_folder, load_wf

parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

# load config file
with open(args.config_filename, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# pull out config values for conciseness
path_config = config["paths"]
key = path_config["key"]
data_config = config['dataParams']
station = data_config["station"]
channel = data_config["channel"]
channel_ID = data_config["channel_ID"]
sampling_rate = data_config["sampling_rate"]

# build path strings
dataH5_name = f'data_{key}.h5'
projectPath = path_config["projectPath"]
pathWF = path_config["pathWF"]
dataH5_path = projectPath + '/H5files/' + dataH5_name
wf_cat_out_path = projectPath + 'wf_cat_out.csv'

if not os.path.isdir(projectPath + '/H5files/'):
    os.mkdir(projectPath + '/H5files/')

# get global catalog
cat = pd.read_csv(path_config["pathCat"])

# get list of waveforms and sort
wf_filelist = [pathWF+x for x in cat["filename"]]
wf_filelist.sort()
print(wf_filelist[0])

if data_config["filetype"] == '.txt':
    wf_test = np.loadtxt(wf_filelist[0])
    lenData = len(wf_test)

else:
    wf_test = obspy.read(wf_filelist[0])
    lenData = len(wf_test[0].data)

# define generator (function)
gen_wf = gen_wf_from_folder(wf_filelist,
                            lenData,
                            data_config["channel_ID"])


# clear old H5 if it exists, or else error will appear
if os.path.exists(dataH5_path):
    os.remove(dataH5_path)

# add catalog and waveforms to H5
evID_keep = [] #list of wfs to keep

with h5py.File(dataH5_path,'a') as h5file:
    global_catalog_group =  h5file.create_group("catalog/global_catalog")

    for col in ["evID", "timestamp", "filename"]:
        if col == "evID":
            global_catalog_group.create_dataset(name=col,data=cat[col])
        else: ## if there are other columns in your catalog
        #that are stings, then you may need to extend conditional statement
        # to use the dtype='S' flag in the next line
            global_catalog_group.create_dataset(name=col,
                                                data=np.array(cat[col],
                                                dtype='S'))

    waveforms_group  = h5file.create_group("waveforms")
    station_group = h5file.create_group(f"waveforms/{station}")
    channel_group = h5file.create_group(f"waveforms/{station}/{channel}")

    dupl_evID = 0 #duplicate event IDs?? not here, sister

    evID_keep = []
    for n, ev in cat.iterrows():
        if n%500==0:
            print(n, '/', len(cat))
        data = load_wf(pathWF+ev["filename"], lenData, channel_ID)
        if data is not None:
            channel_group.create_dataset(name=str(ev["evID"]), data=data)
            evID_keep.append(ev["evID"])
        else:
            print(ev.evID, " not saved")

    """
    for n, data in enumerate(gen_wf):
        if n%500==0:
            print(n, '/', len(wf_filelist))
        channel_group.create_dataset(name= evID, data=data)

    while n <= len(wf_filelist): # not sure a better way to execute this? But it works

        try:   #catch generator "stop iteration" error
            #these all defined in generator at top of script
            data, n = next(gen_wf)

            if n%500==0:
                print(n, '/', len(wf_filelist))
            # if evID not in group, add dataset to wf group
            if evID not in channel_group:
                channel_group.create_dataset(name= evID, data=data)
                evID_keep.append(evID)
            elif evID in channel_group:
                dupl_evID += 1

        except StopIteration: #handle generator error
            break
    """
    processing_group = h5file.create_group(f"{station}/processing_info")
    processing_group.create_dataset(name= "sampling_rate_Hz", data=sampling_rate)#,dtype='S')
    # processing_group.create_dataset(name= "station_info", data=station_info)
    # processing_group.create_dataset(name= "calibration", data=calib)#,dtype='S')
    # processing_group.create_dataset(name= "orig_formata", data=_format)#,dtype='S')
    # processing_group.create_dataset(name= "instr_response", data=instr_response,dtype='S')
    processing_group.create_dataset(name= "lenData", data=lenData)#,dtype='S')

print(dupl_evID, ' duplicate events found and avoided')
print(n + 1 - dupl_evID, ' waveforms loaded')

# save final working catalog to csv
cat_keep_wf = cat[cat['evID'].isin(evID_keep)]

if os.path.exists(wf_cat_out_path):
    os.remove(wf_cat_out_path)

cat_keep_wf.to_csv(wf_cat_out_path, index=False)

print(len(cat_keep_wf), ' events in wf catalog')