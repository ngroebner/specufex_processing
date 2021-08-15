#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:19 2021


example: Parkfield repeaters::




@author: theresasawi
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from functions import dataframe2hdf
from functions.generators import gen_sgram_QC

# command line argument instead of hard coding to config file
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

sgram_config = config["sgramParams"]
nfft = sgram_config["nfft"]
fmin, fmax = sgram_config["fmin"], sgram_config["fmax"]

# build path strings
dataH5_name = f'data_{key}.h5'
projectPath = path_config["projectPath"]
pathWF = path_config["pathWF"]

dataH5_name =  'data_' + path_config["h5name"] #f'data_{key}.hdf5'
dataH5_path = projectPath + 'H5files/' + dataH5_name
SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"] #f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = projectPath + 'H5files/' + SpecUFEx_H5_name
pathWf_cat  = projectPath + 'wf_cat_out.csv'
pathSgram_cat = projectPath + f'sgram_cat_out_{key}.csv'

# get wf catalog
wf_cat = pd.read_csv(pathWf_cat)
evID_list = list(wf_cat.evID)

print('length of event file list: ',len(evID_list))

# get sgram params
with h5py.File(dataH5_path,'r+') as fileLoad:
    # ## sampling rate, Hz
    fs = fileLoad[f"{station}/processing_info"].get('sampling_rate_Hz')[()]
    # ##number of datapoints
    lenData = fileLoad[f"{station}/processing_info"].get('lenData')[()]

##spectrogram parameters, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
nperseg = int(sgram_config["winLen_Sec"]*fs) #datapoints per window segment
noverlap = nperseg*sgram_config["fracOverlap"]  #fraction of window overlapped

#padding must be longer than n per window segment
if nfft < nperseg:
    nfft = nperseg*2
    print("nfft too short; changing to ", nfft)

mode='magnitude'
scaling='spectrum'

# set args for generator
args = {'station':station,
        'channel':channel,
        'fs': fs,
        'lenData': lenData,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'mode': mode,
        'scaling': scaling,
        'fmin': fmin,
        'fmax': fmax}

# put sgrams in h5
### ### ### CREATE GENERATOR ### ### ###
gen_sgram = gen_sgram_QC(key,
                        evID_list=evID_list,
                        dataH5_path = dataH5_path,
                        h5File=fileLoad, #h5 data file
                        trim=True, #trim to min and max freq
                        saveMat=False, #set true to save folder of .mat files
                        sgramOutfile='.', #path to save .mat files
                        **args
                        ) #path to save sgram figures
evID_list_QC_sgram = []

with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:

    n=0
    Nkept=0

    if 'spectrograms' in fileLoad.keys():
        del fileLoad["spectrograms"]

    if 'sgram_normConst' in fileLoad.keys():
        del fileLoad["sgram_normConst"]

    spectrograms_group     = fileLoad.create_group(f"spectrograms")

    sgram_normConst_group  = fileLoad.create_group(f"sgram_normConst")

    while n <= len(evID_list): ## not sure a better way to execute this? But it works
        try:   #catch generator "stop iteration" error
            evID,sgram,fSTFT,tSTFT, normConstant, Nkept,evID_BADones, i = next(gen_sgram) #next() command updates generator
            n = i+1
            evID = str(evID)

            if not evID in spectrograms_group:
                spectrograms_group.create_dataset(name= evID, data=sgram)
                evID_list_QC_sgram.append(np.int64(evID))

            if not evID in sgram_normConst_group:
                sgram_normConst_group.create_dataset(name= evID, data=normConstant)

        except StopIteration: #handle generator error
            break

    print('N events in evID_list_QC_sgram:', len(evID_list_QC_sgram))
    print('N events in evID_BADones:', len(evID_BADones))

    if 'spec_parameters' in fileLoad.keys():
        del fileLoad["spec_parameters"]

    spec_parameters_group  = fileLoad.create_group(f"spec_parameters")
    spec_parameters_group.clear()
    spec_parameters_group.create_dataset(name= 'fs', data=fs)
    spec_parameters_group.create_dataset(name= 'lenData', data=lenData)
    spec_parameters_group.create_dataset(name= 'nperseg', data=nperseg)
    spec_parameters_group.create_dataset(name= 'noverlap', data=noverlap)
    spec_parameters_group.create_dataset(name= 'nfft', data=nfft)
    spec_parameters_group.create_dataset(name= 'mode', data=mode)
    spec_parameters_group.create_dataset(name= 'scaling', data=scaling)
    spec_parameters_group.create_dataset(name= 'fmin', data=fmin)
    spec_parameters_group.create_dataset(name= 'fmax', data=fmax)
    spec_parameters_group.create_dataset(name= 'fSTFT', data=fSTFT)
    spec_parameters_group.create_dataset(name= 'tSTFT', data=tSTFT)

print(evID_list_QC_sgram[0])
print(type(evID_list_QC_sgram[0]))

print(wf_cat['evID'].iloc[0])
print(type(wf_cat['evID'].iloc[0]))

# merge catalogs
print(len(wf_cat))
cat_keep_sgram = wf_cat[wf_cat['evID'].isin(evID_list_QC_sgram)]
print(len(cat_keep_sgram))
#print(cat_keep_sgram)

try:
#   cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)
    cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)
except:
   pass

if os.path.exists(pathSgram_cat):
    os.remove(pathSgram_cat)

cat_keep_sgram.to_csv(pathSgram_cat)

# save local catalog to original datafile
with h5py.File(dataH5_path,'a') as h5file:
    if f'catalog/cat_by_sta/{station}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{station}"]
    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{station}")
    dataframe2hdf(cat_keep_sgram, catalog_sta_group)

# save local catalog to new ML datafile
with h5py.File(SpecUFEx_H5_path,'a') as h5file:
    if f'catalog/cat_by_sta/{station}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{station}"]
    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{station}")
    dataframe2hdf(cat_keep_sgram, catalog_sta_group)
