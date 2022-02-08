#!python3
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

import h5py
import pandas as pd
import yaml

from specufex_processing.preprocessing import dataframe2hdf,normalize_waveform
from specufex_processing.utils import _overwrite_group_if_exists
from specufex_processing.distance import utils_distance as udst
from specufex_processing.clustering import functions_clustering as funclust
from specufex_processing.plotting import Basic_config

import pdb

Basic_config.set_plot_prop()

if __name__ == "__main__":
    # command line argument instead of hard coding to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", help="Path to configuration file.")
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # pull out config values for conciseness
    path_config = config["paths"]
    key = path_config["key"]

    data_config = config['dataParams']

    factor_spec_sampling = int(data_config['sampling_rate']* config['sgramParams']['winLen_Sec'])
    station = data_config["station"]
    channel = data_config["channel"]
    channel_ID = data_config["channel_ID"]
    sampling_rate = data_config["sampling_rate"]

    # build path strings
    dataH5_name = f'data_{key}.h5'
    projectPath = path_config["projectPath"]
    os.system(f' cp {args.config_filename} {projectPath}/')
    dataH5_path = os.path.join(projectPath,'H5files/', dataH5_name)
    SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"]
    SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)
    pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')

    path_dist_matrix_base = os.path.join(config['paths']['projectPath'],"distance_matrices")
    os.makedirs(path_dist_matrix_base,exist_ok=True)

    sgram_catPath = os.path.join(projectPath,f'sgram_cat_out_{key}.csv')
    cat0 = pd.read_csv(sgram_catPath)
    cat0['index_counter'] = cat0.index.values
    cat0['ev_ID'] = cat0['ev_ID'].astype(str)
    cat0.index = cat0['ev_ID']

    distance_params = config['distanceParams']
    overwrite_distance = distance_params['overwrite_distance']
    distance_measure_name_list = ['_L1_distance','_L2_distance','_cos_distance','_corr_distance']
    distance_measure_name_list_val = ['l1','l2','cosine','correlation']

    cluster_yn = distance_params['do_clustering_dist']
    cluster_measure = distance_params['clustering_measure']
    cluster_numK = distance_params['K_vals']

    ### Load the waveforms ...
    X = []
    with h5py.File(dataH5_path,'r') as fileLoad:
        evID_waves = []
        for evID in fileLoad[f'waveforms/{station}/{channel}']:
            specMat = fileLoad[f'waveforms/{station}/{channel}'].get(evID)[:]
            X.append(specMat)
            evID_waves.append(evID)

        X = np.array(X)

    wave_norm = distance_params['waveform_normalization']
    X_norm = normalize_waveform(X,norm_scale = wave_norm,save_plot=False)
    ##############################################################

    if distance_params['distance_waveforms'] :
        print('Loading the waveforms ... ')
        path_dist_matrix_wave = os.path.join(path_dist_matrix_base,"waveforms")
        os.makedirs(path_dist_matrix_wave,exist_ok=True)
        waveform_size = X.shape[1]
        distance_params['shift_allowed'] = int(waveform_size*distance_params['frac_shift_allowed'])

        if distance_params['distance_nonNorm'] :
            prefix_name = path_dist_matrix_wave+'/Non_normalized_waveform'
            os.makedirs(prefix_name,exist_ok=True)
            prefix_name = prefix_name+'/Non_normalized_waveform'

            udst.calculate_distances_all(cat0,evID_waves,X,distance_params,prefix_name,
                                        distance_measure_name_list,
                                        distance_measure_name_list_val,
                                        use_cross_corr=True,
                                        make_plots=distance_params['make_dist_plots'],
                                        cluster_yn = cluster_yn,
                                        cluster_measure = cluster_measure,
                                        cluster_numK = cluster_numK,
                                        norm_waveforms = X_norm,wave_id=evID_waves)

        wave_norm = distance_params['waveform_normalization']
        prefix_name = path_dist_matrix_wave+f'/{wave_norm}_normalized_waveform'
        os.makedirs(prefix_name,exist_ok=True)
        X_norm = normalize_waveform(X,norm_scale = wave_norm,save_plot=True,
                                    path_to_save = prefix_name +f"/Normalized_{wave_norm}")

        prefix_name = prefix_name+f'/{wave_norm}_normalized_waveform'
        udst.calculate_distances_all(cat0,evID_waves,X_norm,distance_params,prefix_name,
                                    distance_measure_name_list,
                                    distance_measure_name_list_val,
                                    use_cross_corr=True,
                                    make_plots=distance_params['make_dist_plots'],
                                        cluster_yn = cluster_yn,
                                        cluster_measure = cluster_measure,
                                        cluster_numK = cluster_numK,
                                        norm_waveforms = X_norm,wave_id=evID_waves)
        plt.close('all')
        del X


    if distance_params['distance_spectra'] :
        print('Loading the spectra ... ')
        path_dist_matrix_spec = os.path.join(path_dist_matrix_base,"spectrum")
        os.makedirs(path_dist_matrix_spec,exist_ok=True)

        X = []
        with h5py.File(SpecUFEx_H5_path,'r') as fileLoad:
            evID_hdf5 = []
            for evID in fileLoad['spectrograms/transformed_spectrograms']:
                specMat = fileLoad['spectrograms/transformed_spectrograms'].get(evID)[:]
                X.append(specMat)
                evID_hdf5.append(evID)

        X = np.array(X)
        waveform_size = X.shape[2]
        distance_params['shift_allowed'] = int(waveform_size*0.5)

        prefix_name = path_dist_matrix_spec+'/normalized_Spectrum'
        os.makedirs(prefix_name,exist_ok=True)
        prefix_name = prefix_name+'/normalized_Spectrum'
        udst.calculate_distances_all(cat0,evID_hdf5,X,distance_params,prefix_name,
                                    distance_measure_name_list,
                                    distance_measure_name_list_val,
                                    use_cross_corr=True,use_2d=True,
                                    make_plots=distance_params['make_dist_plots'],
                                    cluster_yn = cluster_yn,
                                    cluster_measure = cluster_measure,
                                    cluster_numK = cluster_numK,
                                    norm_waveforms = X_norm,wave_id=evID_waves,factor_spec_sampling=factor_spec_sampling)
        plt.close('all')
        del X

        if distance_params['distance_spectra_nonNorm'] :
            X = []
            with h5py.File(SpecUFEx_H5_path,'r') as fileLoad:
                evID_hdf5 = []
                for evID in fileLoad['spectrograms/raw_spectrograms']:
                    specMat = fileLoad['spectrograms/raw_spectrograms'].get(evID)[:]
                    X.append(specMat)
                    evID_hdf5.append(evID)

            X = np.array(X)
            waveform_size = X.shape[2]
            distance_params['shift_allowed'] = int(waveform_size*0.5)

            prefix_name = path_dist_matrix_spec+'/Raw_Spectrum'
            os.makedirs(prefix_name,exist_ok=True)
            prefix_name = prefix_name+'/Raw_Spectrum'

            udst.calculate_distances_all(cat0,evID_hdf5,X,distance_params,prefix_name,
                                        distance_measure_name_list,
                                        distance_measure_name_list_val,
                                        use_cross_corr=True,use_2d=True,
                                        make_plots=distance_params['make_dist_plots'],
                                        cluster_yn = cluster_yn,
                                        cluster_measure = cluster_measure,
                                        cluster_numK = cluster_numK,
                                        norm_waveforms = X_norm,wave_id=evID_waves,factor_spec_sampling=factor_spec_sampling)

            plt.close('all')
            del X

    if distance_params['distance_fingerprint'] :
        print('Loading the Fingerprints ... ')
        path_dist_matrix_fing = os.path.join(path_dist_matrix_base,"fingerprint")
        os.makedirs(path_dist_matrix_fing,exist_ok=True)

        X = []
        with h5py.File(SpecUFEx_H5_path,'r') as fileLoad:
            evID_hdf5 = []
            for evID in fileLoad['fingerprints']:
                specMat = fileLoad['fingerprints'].get(evID)[:]
                X.append(specMat)
                evID_hdf5.append(evID)

        X = np.array(X)
        waveform_size = X.shape[2]
        distance_params['shift_allowed'] = int(waveform_size*0.5)

        prefix_name = path_dist_matrix_fing+'/fingerprints'

        udst.calculate_distances_all(cat0,evID_hdf5,X,distance_params,prefix_name,
                                        distance_measure_name_list,
                                        distance_measure_name_list_val,
                                        use_cross_corr=True,use_2d=True,
                                        make_plots=distance_params['make_dist_plots'],
                                        cluster_yn = cluster_yn,
                                        cluster_measure = cluster_measure,
                                        cluster_numK = cluster_numK,
                                        norm_waveforms = X_norm,wave_id=evID_waves)

        plt.close('all')
        del X
