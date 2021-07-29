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

from functions.setParams import setParams,setSgramParams
from functions.generators import gen_sgram_QC

# ============================================
# STUFF to change when we go to config.py method
#%% load project variables: names and paths

parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

# pick the operating system, for pandas.to_csv
OSflag = 'linux'
#OSflag = 'mac'
# =====================================================

# TODO: convert to config file method
pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)


pathCatWF = pathCat


dataH5_name = f'data_{key}.hdf5'
dataH5_path = pathProj + '/H5files/' + dataH5_name


SpecUFEx_H5_name = f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = pathProj + '/H5files/' + SpecUFEx_H5_name

# ## for testing
# sgramMatOut = pathProj + 'matSgrams/'


pathWf_cat  = pathProj + 'wf_cat_out.csv'
pathSgram_cat = pathProj + f'sgram_cat_out_{key}.csv'


#%% get wf catalog

wf_cat = pd.read_csv(pathWf_cat)
evID_list = list(wf_cat.event_ID)

print('length of event file list: ',len(evID_list))

#%% get sgram params
fmin, fmax, winLen_Sec, fracOverlap, nfft = setSgramParams(key)

with h5py.File(dataH5_path,'r+') as fileLoad:

    # ## sampling rate, Hz
    fs = fileLoad[f"{station}/processing_info"].get('sampling_rate_Hz')[()]

    # ##number of datapoints
    lenData = fileLoad[f"{station}/processing_info"].get('lenData')[()]

##spectrogram parameters, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
nperseg = int(winLen_Sec*fs) #datapoints per window segment
noverlap = nperseg*fracOverlap  #fraction of window overlapped

#padding must be longer than n per window segment
if nfft < nperseg:
    nfft = nperseg*2
    print("nfft too short; changing to ", nfft)


mode='magnitude'
scaling='spectrum'


#%% set args for generator

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



#%% put sgrams in h5
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



#%%

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

print(wf_cat['event_ID'].iloc[0])
print(type(wf_cat['event_ID'].iloc[0]))

#%% merge catalogs
print(len(wf_cat))
cat_keep_sgram = wf_cat[wf_cat['event_ID'].isin(evID_list_QC_sgram)]
print(len(cat_keep_sgram))
#print(cat_keep_sgram)


try:
#   cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)
    cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)
except:
   pass

if os.path.exists(pathSgram_cat):
    os.remove(pathSgram_cat)

print('formatting CSV catalog for ',OSflag)
if OSflag=='linux':
    cat_keep_sgram.to_csv(pathSgram_cat,line_terminator='\n')
elif OSflag=='mac':
    cat_keep_sgram.to_csv(pathSgram_cat)


'''
NOT SURE WE NEED THIS--- test removing it on mac and linux
line_terminatorstr, optional
The newline character or character sequence to use in the output file.
Defaults to os.linesep, which depends on the OS
in which this method is called (‘\n’ for linux, ‘\r\n’ for Windows, i.e.).
so easiest fix may be to try to insert line_terminator='\n'
in lines ~180 or 188 in 1_ and 2_.py
? wherever “to_csv” is
'''
#%% save local catalog to original datafile

with h5py.File(dataH5_path,'a') as h5file:

    if f'catalog/cat_by_sta/{station}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{station}"]

    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{station}")


    for col in cat_keep_sgram.columns:



        if col == 'datetime':
            catalog_sta_group.create_dataset(name='datetime',data=np.array(cat_keep_sgram['datetime'],dtype='S'))

        else:
            exec(f"catalog_sta_group.create_dataset(name='{col}',data=cat_keep_sgram.{col})")




#%% save local catalog to new ML datafile

with h5py.File(SpecUFEx_H5_path,'a') as h5file:

    if f'catalog/cat_by_sta/{station}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{station}"]

    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{station}")


    for col in cat_keep_sgram.columns:



        if col == 'datetime':
            catalog_sta_group.create_dataset(name='datetime',data=np.array(cat_keep_sgram['datetime'],dtype='S'))

        else:
            exec(f"catalog_sta_group.create_dataset(name='{col}',data=cat_keep_sgram.{col})")








#%%




#%%









#%%







#%%
