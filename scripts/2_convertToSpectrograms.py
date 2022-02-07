#!python3

import argparse
import os

import h5py
import pandas as pd
import yaml

from specufex_processing.preprocessing import dataframe2hdf
from specufex_processing.preprocessing.spectrogram import create_spectrograms, pad_spects
from specufex_processing.utils import _overwrite_group_if_exists
import pdb

if __name__ == "__main__":

    # load project variables: names and paths
    #args = parser.parse_args(

    # command line argument instead of hard coding to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", help="Path to configuration file.")
    args = parser.parse_args()

    with open(args.config_filename, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    config_path = config['paths']
    config_dataparams = config['dataParams']
    config_sgram = config["sgramParams"]

    pathCatWF = config_path['pathCat']
    waveform_path = config_path['pathWF']
    projectPath = config_path["projectPath"]
    spectrogram_H5_name = 'spectrograms.h5'
    spectrogram_H5_path = os.path.join(projectPath,"data","spectrograms",spectrogram_H5_name)
    key = config_path["key"]
    dataH5_name = f'data_{key}.h5'
    dataH5_path = os.path.join(projectPath,'H5files/', dataH5_name)
    SpecUFEx_H5_name = 'SpecUFEx_' + config_path["h5name"]
    SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)

    station  = config_dataparams["station"]
    channel = config_dataparams["channel"]

    os.makedirs(os.path.join(config['paths']['projectPath'],"data","spectrograms"),exist_ok=True)

    pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')
    pathSgram_cat = os.path.join(projectPath,f'sgram_cat_out_{key}.csv')
    # get wf catalog
    wf_cat = pd.read_csv(pathWf_cat)
    evID_list = list(wf_cat.ev_ID)

    # get sgram params
    fmin = config_sgram['fmin']
    fmax = config_sgram['fmax']
    norm_waveforms = config_sgram["norm_waveforms"]
    winLen_Sec = config_sgram['winLen_Sec']
    fracOverlap = config_sgram['fracOverlap']
    nfft = config_sgram['nfft']

    evIDs, spects, raw_spects, spectmaker = create_spectrograms(
                        dataH5_path,
                        station,
                        channel,
                        winLen_Sec,
                        fracOverlap,
                        nfft,
                        fmin,
                        fmax,
                        norm_waveforms=norm_waveforms
    )
    # pad short spectrograms with zeros

    spects = pad_spects(spects)
    raw_spects = pad_spects(raw_spects)

    print("Spectrograms created.")
    print("Size of spectrogram", spects[0].shape)

    spectmaker.save2hdf5(spects, raw_spects, evIDs, SpecUFEx_H5_path)

    # merge catalogs
    #pdb.set_trace()
    cat_keep_sgram = wf_cat[wf_cat['ev_ID'].astype(str).isin(evIDs)]

    try:
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

        # save config file values used in this step
        pathattr = _overwrite_group_if_exists(h5file, "path_attrs")
        for attr in config_path.keys():
            pathattr.attrs[attr] = config_path[attr]

        dataattr = _overwrite_group_if_exists(h5file, "dataparam_attrs")
        for attr in config_dataparams.keys():
            dataattr.attrs[attr] = config_dataparams[attr]

        sgramattr = _overwrite_group_if_exists(h5file, "sgram_attrs")
        for attr in config_sgram.keys():
            sgramattr.attrs[attr] = config_sgram[attr]
