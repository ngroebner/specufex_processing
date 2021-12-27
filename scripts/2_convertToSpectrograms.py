#!python3

import argparse
import os

import h5py
import pandas as pd
import yaml

from specufex_processing.preprocessing import dataframe2hdf
from specufex_processing.preprocessing.spectrograms import create_spectrograms, pad_spects


if __name__ == "__main__":

    # load project variables: names and paths
    #args = parser.parse_args(

    # command line argument instead of hard coding to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", help="Path to configuration file.")
    args = parser.parse_args()

    with open(args.config_filename, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    pathCatWF = config['paths']['pathCat']
    waveform_path = config['paths']['pathWF']
    projectPath = config['paths']["projectPath"]
    spectrogram_H5_name = 'spectrograms.h5'
    spectrogram_H5_path = os.path.join(projectPath,"data","spectrograms",spectrogram_H5_name)
    key = config["paths"]["key"]
    dataH5_name = f'data_{key}.h5'
    dataH5_path = projectPath + '/H5files/' + dataH5_name
    SpecUFEx_H5_name = 'SpecUFEx_' + config['paths']["h5name"]
    SpecUFEx_H5_path = projectPath + 'H5files/' + SpecUFEx_H5_name

    station  = config["dataParams"]["station"]
    channel = config["dataParams"]["channel"]

    os.makedirs(os.path.join(config['paths']['projectPath'],"data","spectrograms"),exist_ok=True)

    pathWf_cat  = projectPath + 'wf_cat_out.csv'
    pathSgram_cat = projectPath + f'sgram_cat_out_{key}.csv'
    # get wf catalog
    wf_cat = pd.read_csv(pathWf_cat)
    evID_list = list(wf_cat.ev_ID)

    # get sgram params
    fmin = config['sgramParams']['fmin']
    fmax = config['sgramParams']['fmax']
    winLen_Sec = config['sgramParams']['winLen_Sec']
    fracOverlap = config['sgramParams']['fracOverlap']
    nfft = config['sgramParams']['nfft']

    print(f"fmin:{fmin} fmax: {fmax}")

    evIDs, spects, spectmaker = create_spectrograms(
                        dataH5_path,
                        station,
                        channel,
                        winLen_Sec,
                        fracOverlap,
                        nfft,
                        fmin,
                        fmax)

    # pad short spectrograms with zeros

    spects = pad_spects(spects)

    print("Spectrograms created.")

    print("Size of spectrogram", spects[0].shape)

    spectmaker.save2hdf5(spects, evIDs, spectrogram_H5_path)

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
        cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)
    except:
        pass

    if os.path.exists(pathSgram_cat):
        os.remove(pathSgram_cat)

    cat_keep_sgram.to_csv(pathSgram_cat)

    ## I'm not sure what these do

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
