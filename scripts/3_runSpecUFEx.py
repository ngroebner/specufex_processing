#!python3
# -*- coding: utf-8 -*-


import argparse
import os

import h5py
import numpy as np
import time
import yaml

from specufex import BayesianNonparametricNMF, BayesianHMM
from specufex_processing.utils import (
    _overwrite_dataset_if_exists,
    _overwrite_group_if_exists
)

t_start = time.time()

# command line argument instead of hard coding to config file
parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

with open(args.config_filename, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print("Could not read config file.", exc)
        exit()

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

dataH5_name =  os.path.join('data_', path_config["h5name"])
dataH5_path = os.path.join(projectPath, 'H5files/', dataH5_name)
SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"]
SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)
pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')

X = []

with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    for evID in fileLoad['spectrograms/transformed_spectrograms']:
        specMat = fileLoad['spectrograms/transformed_spectrograms'].get(evID)[:]
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

print(f"NMF time: {t_nmf1-t_nmf0}")

# save the model using it's own machinery
print("Saving NMF model and data")
nmf.save(os.path.join(projectPath, 'H5files/', "nmf.h5"), overwrite=True)

# save model parameters and calculated ACMs to the specufex data
with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    if "SpecUFEx_output" in fileLoad:
        out_group = fileLoad["SpecUFEx_output"]
    else:
        out_group = fileLoad.create_group("SpecUFEx_output")
    if "model_parameters" in fileLoad:
        model_group = fileLoad["model_parameters"]
    else:
        model_group = fileLoad.create_group("model_parameters")

    _overwrite_dataset_if_exists(out_group, name="ACM_gain", data=nmf.gain)
    _overwrite_dataset_if_exists(model_group, name='EW',data=nmf.EW)
    _overwrite_dataset_if_exists(model_group, name='EA',data=nmf.EA)
    _overwrite_dataset_if_exists(model_group, name='ElnWA',data=nmf.ElnWA)
    ACM_group = _overwrite_group_if_exists(fileLoad, "ACM")
    for i, evID in enumerate(fileLoad['spectrograms/transformed_spectrograms']):
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
    fp_group = _overwrite_group_if_exists(fileLoad, "fingerprints")

    # save fingerprints
    for i, evID in enumerate(fileLoad['spectrograms/transformed_spectrograms']):
        fp_group.create_dataset(name= evID, data=fingerprints[i])

    # save the A and gam vectors
    As_group = _overwrite_group_if_exists(fileLoad, "SpecUFEx_output/As")
    STM_group = _overwrite_group_if_exists(fileLoad, "STM")

    for i, evID in enumerate(fileLoad['fingerprints']):
         As_group.create_dataset(name=evID,data=As[i])
         STM_group.create_dataset(name=evID,data=gams[i])

    _overwrite_dataset_if_exists(fileLoad, "model_parameters/EB", data=hmm.EB)
    ## # # delete probably ! gain_group                   = fileLoad.create_group("SpecUFEX_output/gain")
    #RMM_group                    = fileLoad.create_group("SpecUFEX_output/RMM")

