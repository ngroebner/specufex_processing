#!python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:02:51 2021

@author: theresasawi
"""

import argparse
import os

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import yaml

from specufex import BayesianNonparametricNMF, BayesianHMM

t_start = time.time()

# command line argument instead of hard coding to config file
parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

with open(args.config_filename, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print("F you!", exc)

variable_list = []
value_list = []

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

dataH5_name =  os.path.join('data_', path_config["h5name"]) #f'data_{key}.hdf5'
dataH5_path = os.path.join(projectPath, 'H5files/', dataH5_name)
SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"] #f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)
pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')
pathSgram_cat = os.path.join(projectPath, f'sgram_cat_out_{key}.csv')
sgramMatOut = os.path.join(projectPath, 'matSgrams/')## for testing

sgram_cat = pd.read_csv(pathSgram_cat)

# load spectrograms
X = []

with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    for evID in fileLoad['spectrograms']:
        specMat = fileLoad['spectrograms'].get(evID)[:]
        X.append(specMat)

    X = np.array(X)

# ============================================================
# Running SpecUFEx
# ============================================================

specparams = config["specufexParams"]

t_nmf0 = time.time()
print('Running NMF')
nmf = BayesianNonparametricNMF(X.shape)
for i in range(specparams["nmf_nbatch"]):
    # pick random sample
    print(f"Batch {i}")
    sample = np.random.choice(
        X.shape[0],
        specparams["nmf_batchsz"],
        replace=False
    )
    nmf.fit(X[sample], verbose=1)

print("Calculating ACMs.")

Vs = nmf.transform(X)
t_nmf1 = time.time()
# print how long it took?
print(f"NMF time: {t_nmf1-t_nmf0}")

# save the model using it's own machinery
print("Saving NMF model and data")

nmf.save(os.path.join(projectPath, 'H5files/', "nmf.h5"), overwrite=True)

# save model parameters and calculated ACMs to the specufex data
with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    print(SpecUFEx_H5_path)
    print(fileLoad.keys())
    if "SpecUFEx_output" in fileLoad:
        out_group = fileLoad["SpecUFEx_output"]
    else:
        out_group = fileLoad.create_group("SpecUFEx_output")
    out_group.create_dataset(name="ACM_gain", data=nmf.gain)
    out_group.create_dataset(name='EW',data=nmf.EW)
    out_group.create_dataset(name='EA',data=nmf.EA)
    ACM_group = out_group.create_group("ACM") # activation coefficient matrix
    for i, evID in enumerate(fileLoad['spectrograms']):
        ACM_group.create_dataset(name=evID, data=Vs[i])

t_hmm0 = time.time()
print('Running HMM')
hmm = BayesianHMM(nmf.num_pat, nmf.gain)
for i in range(specparams["hmm_nbatch"]):
    print(f"Batch {i}")
    sample = np.random.choice(
        Vs.shape[0],
        specparams["nmf_batchsz"],
        replace=False
    )
    hmm.fit(Vs[sample], verbose=1)

print("Calculating fingerprints")
fingerprints, As, gams = hmm.transform(Vs)
t_hmm1 = time.time()
print(f"HMM time: {t_hmm1-t_hmm0}")

print('Saving HMM model and data')
hmm.save(os.path.join(projectPath, 'H5files/', "hmm.h5"), overwrite=True)
with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:


    ##fingerprints are top folder
    if 'fingerprints' in fileLoad.keys():
        del fileLoad["fingerprints"]
    fp_group = fileLoad.create_group('fingerprints')

    # write fingerprints: ===============================
    for i, evID in enumerate(fileLoad['spectrograms']):
        fp_group.create_dataset(name= evID, data=fingerprints[i])

    # write the SpecUFEx out: ===========================
    As_group = fileLoad.create_group("SpecUFEx_output/As")
    STM_group = fileLoad.create_group("SpecUFEx_output/STM")

    for i, evID in enumerate(fileLoad['spectrograms']):
         As_group.create_dataset(name=evID,data=As[i])
         STM_group.create_dataset(name=evID,data=gams[i]) #STM

    fileLoad.create_dataset(name="SpecUFEx_output/EB", data=hmm.EB)
    ## # # delete probably ! gain_group                   = fileLoad.create_group("SpecUFEX_output/gain")
    #RMM_group                    = fileLoad.create_group("SpecUFEX_output/RMM")

