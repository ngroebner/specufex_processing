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
from functions.spectrogram import create_spectrograms, pad_spects, save_spectrograms



if __name__ == "__main__":

    # load project variables: names and paths
    #args = parser.parse_args(

    # command line argument instead of hard coding to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", help="Path to configuration file.")
    args = parser.parse_args()

    with open("params.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)['spectrograms']

    pathCatWF = config['pathCat']
    waveform_path = config['waveform_path']
    spectrogram_H5_name = 'spectrograms.h5'
    spectrogram_H5_path = os.path.join(config['pathProj'],"data","spectrograms",spectrogram_H5_name)

    os.makedirs(os.path.join(config['pathProj'],"data","spectrograms"),exist_ok=True)

    pathWf_cat  = pathCatWF + 'wf_cat_out.csv'
    pathSgram_cat = config['pathProj'] + f'sgram_cat_out_{config["name"]}.csv'
    # get wf catalog
    wf_cat = pd.read_csv(pathWf_cat)
    evID_list = list(wf_cat.evID)

    # get sgram params
    fmin = config['fmin']
    fmax = config['fmax']
    winLen_Sec = config['winLen_Sec']
    fracOverlap = config['fracOverlap']
    nfft = config['nfft']

    print(f"fmin:{fmin} fmax: {fmax}")

    evIDs, spects = create_spectrograms(
                        waveform_path,
                        winLen_Sec,
                        fracOverlap,
                        nfft,
                        fmin,
                        fmax)

    # pad short spectrograms with zeros

    spects = pad_spects(spects)

    print("Spectrograms created.")

    print("Size of spectrogram", spects.shape)

    save_spectrograms(spects, evIDs, spectrogram_H5_path)


#print(evID_list_QC_sgram[0])
#print(type(evID_list_QC_sgram[0]))

print(wf_cat['ev_ID'].iloc[0])
print(type(wf_cat['ev_ID'].iloc[0]))

# merge catalogs
print(len(wf_cat))
cat_keep_sgram = wf_cat[wf_cat['ev_ID'].isin(evIDs)]
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
