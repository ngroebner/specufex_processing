#!python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:02:51 2021

@author: theresasawi
"""

import argparse

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml

from specufex import BayesianNonparametricNMF, BayesianHMM

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

dataH5_name =  'data_' + path_config["h5name"] #f'data_{key}.hdf5'
dataH5_path = projectPath + 'H5files/' + dataH5_name
SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"] #f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = projectPath + 'H5files/' + SpecUFEx_H5_name
pathWf_cat  = projectPath + 'wf_cat_out.csv'
pathSgram_cat = projectPath + f'sgram_cat_out_{key}.csv'
sgramMatOut = projectPath + 'matSgrams/'## for testing

sgram_cat = pd.read_csv(pathSgram_cat)

#%% get spectrograms from H5

# a NUMPY WAY (can certainly be cleaned/tightened up-- very verbose as written:)
# and I got the shape wrong.. the list .append() adds to axis=0, not 2.

# with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
#     count = 0
#     for evID in fileLoad['spectrograms']:
#         count += 1
#         if count==1:
#             specMat = fileLoad['spectrograms'].get(evID)[:]
#             sg_shp = np.shape(specMat)
#
#
# X = np.empty((sg_shp[0],sg_shp[1],count))
#
# with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
#     i_count = 0
#     for evID in fileLoad['spectrograms']:
#         specMat = fileLoad['spectrograms'].get(evID)[:]
#         Xis = specMat
#         if np.shape(Xis) != sg_shp:
#             print(evID, i_count,': SHAPE IS DIFFERENT!')
#
#         X[:,:,i_count] = Xis
#         i_count += 1
#         if i_count%500 == 0:
#             print(i_count)

# ================
# LIST WAY (how it was written before):
X = []

with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    for evID in fileLoad['spectrograms']:
        specMat = fileLoad['spectrograms'].get(evID)[:]
        X.append(specMat)

    X = np.array(X)

# ================
# print(np.shape(X))
# print(X[:,:,-1])



#%% ============================================================
# Running SpecUFEx
#%% ============================================================

specparams = config["specufexParams"]

print('Running NMF')
nmf = BayesianNonparametricNMF(X.shape)
for i in range(specparams["nmf_nbatch"]):
    # pick random sample
    print(f"Batch {i}")
    sample = np.random.choice(X.shape[0], specparams["nmf_batchsz"])
    nmf.fit(X[sample], verbose=1)

Vs = nmf.transform(X)
# print how long it took

#%%
print('Running HMM')
hmm = BayesianHMM(nmf.num_pat, nmf.gain)
for i in range(specparams["hmm_nbatch"]):
    print(f"Batch {i}")
    sample = np.random.choice(Vs.shape[0], specparams["nmf_batchsz"])
    hmm.fit(Vs)

fingerprints, As, gams = hmm.transform(Vs)

#print(fingerprints[0])

# show a fingerprint if you want to .. but not useful for running remotely..
#plt.imshow(fingerprints[0])
#plt.show()
#%%
#%%
#%%

# =============================================================================
# save output to H5
# =============================================================================
print('writing all output to h5')
with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:


    ##fingerprints are top folder
    if 'fingerprints' in fileLoad.keys():
        del fileLoad["fingerprints"]
    fp_group = fileLoad.create_group('fingerprints')

    if 'SpecUFEX_output' in fileLoad.keys():
        del fileLoad["SpecUFEX_output"]
    out_group = fileLoad.create_group("SpecUFEX_output")

    # write fingerprints: ===============================
    for i, evID in enumerate(fileLoad['spectrograms']):
        fp_group.create_dataset(name= evID, data=fingerprints[i])
        #ACM_group.create_dataset(name=evID,data=As[i]) #ACM
        #STM_group.create_dataset(name=evID,data=gam[i]) #STM

    # write the SpecUFEx out: ===========================
    # maybe include these, but they are not yet tested.
    #ACM_group = fileLoad.create_group("SpecUFEX_output/ACM")
    #STM_group = fileLoad.create_group("SpecUFEX_output/STM")

    # for i, evID in enumerate(fileLoad['spectrograms']):
    #     ACM_group.create_dataset(name=evID,data=As[i]) #ACM
    #     STM_group.create_dataset(name=evID,data=gam[i]) #STM

    gain_group = fileLoad.create_group("SpecUFEX_output/ACM_gain")
    W_group                      = fileLoad.create_group("SpecUFEX_output/W")
    EB_group                     = fileLoad.create_group("SpecUFEX_output/EB")
    ## # # delete probably ! gain_group                   = fileLoad.create_group("SpecUFEX_output/gain")
    #RMM_group                    = fileLoad.create_group("SpecUFEX_output/RMM")

    W_group.create_dataset(name='W',data=nmf.EW)
    EB_group.create_dataset(name=evID,data=hmm.EB)
    gain_group.create_dataset(name='gain',data=nmf.gain) #same for all data
    # RMM_group.create_dataset(name=evID,data=RMM)
