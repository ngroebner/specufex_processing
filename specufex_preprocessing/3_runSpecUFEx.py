#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:02:51 2021

@author: theresasawi
"""

import h5py
import numpy as np
import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('functions/')
from setParams import setParams
# from generators import gen_sgram_QC

import tables
tables.file._open_files.close_all()
from specufex import BayesianNonparametricNMF, BayesianHMM


#%% load project variables: names and paths

key = sys.argv[1]
#

#%%
### do not change these ###

pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)



dataH5_name = f'data_{key}.hdf5'
dataH5_path = pathProj + '/H5files/' + dataH5_name
SpecUFEx_H5_name = f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = pathProj + '/H5files/' + SpecUFEx_H5_name
sgramMatOut = pathProj + 'matSgrams/'## for testing
pathWf_cat  = pathProj + 'wf_cat_out.csv'

# Why is this here-- it is not being used.
pathSgram_cat = pathProj + f'sgram_cat_out_{key}.csv'

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
        Xis = specMat
        X.append(Xis)

    X = np.array(X)

# ================
print(np.shape(X))
print(X[:,:,-1])



#%% ============================================================
# Running SpecUFEx
#%% ============================================================

print('Running NMF')
nmf = BayesianNonparametricNMF(X.shape)
nmf.fit(X, verbose=1)
Vs = nmf.transform(X)
# print how long it took

#%%
print('Running HMM')
hmm = BayesianHMM(nmf.num_pat, nmf.gain)
hmm.fit(Vs)
fingerprints, As, gams = hmm.transform(Vs)

print(fingerprints[0])

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
