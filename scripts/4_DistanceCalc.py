#!python3
import argparse
import os
import numpy as np

import h5py
import pandas as pd
import yaml

from specufex_processing.preprocessing import dataframe2hdf,normalize_waveform
from specufex_processing.utils import _overwrite_group_if_exists
from specufex_processing.distance import calc_corrmatrix,calc_distmatrix_Basic_Distances,corr_distance_1D
from specufex_processing.distance import utils_distance as udst
import pdb

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
    station = data_config["station"]
    channel = data_config["channel"]
    channel_ID = data_config["channel_ID"]
    sampling_rate = data_config["sampling_rate"]

    # build path strings
    dataH5_name = f'data_{key}.h5'
    projectPath = path_config["projectPath"]

    dataH5_path = os.path.join(projectPath,'H5files/', dataH5_name)
    SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"]
    SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)
    pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')

    path_dist_matrix_base = os.path.join(config['paths']['projectPath'],"distance_matrices")
    os.makedirs(path_dist_matrix_base,exist_ok=True)

    distance_params = config['distanceParams']
    overwrite_distance = distance_params['overwrite_distance']
    distance_measure_name_list = ['_L1_distance','_L2_distance','_cos_distance','_corr_distance']
    distance_measure_name_list_val = ['l1','l2','cosine','correlation']

    if distance_params['distance_waveforms'] :
        print('Loading the waveforms ... ')
        path_dist_matrix_wave = os.path.join(path_dist_matrix_base,"waveforms")
        os.makedirs(path_dist_matrix_wave,exist_ok=True)

        ### Calculate the distance matrices for the waveforms
        X = []
        with h5py.File(dataH5_path,'r') as fileLoad:
            for evID in fileLoad[f'waveforms/{station}/{channel}']:
                specMat = fileLoad[f'waveforms/{station}/{channel}'].get(evID)[:]
                X.append(specMat)

            X = np.array(X)
            waveform_size = X.shape[1]
        distance_params['shift_allowed'] = int(waveform_size*distance_params['frac_shift_allowed'])

        if distance_params['distance_nonNorm'] :
            prefix_name = path_dist_matrix_wave+'/Non_normalized_waveform'
            udst.calculate_distances_all(X,distance_params,prefix_name,
                                        distance_measure_name_list,
                                        distance_measure_name_list_val,
                                        use_cross_corr=True)

        wave_norm = distance_params['waveform_normalization']
        X_norm = normalize_waveform(X,norm_scale = wave_norm,save_plot=True,
                                    path_to_save = path_dist_matrix_wave +f"/Normalized_{wave_norm}")
        prefix_name = path_dist_matrix_wave+f'/{wave_norm}_normalized_waveform'
        udst.calculate_distances_all(X_norm,distance_params,prefix_name,
                                    distance_measure_name_list,
                                    distance_measure_name_list_val,
                                    use_cross_corr=True)
        del X, X_norm





    with h5py.File(dataH5_path,'a') as fileLoad:
        for evID in fileLoad['spectrograms/transformed_spectrograms']:
            specMat = fileLoad['spectrograms/transformed_spectrograms'].get(evID)[:]
            X.append(specMat)

        X = np.array(X)

    # ============================================================
    # Running SpecUFEx
    # ============================================================



    distance_name = 'dist_matrix.npy'
    spectrogram_H5_path = os.path.join(projectPath,"data","spectrograms",spectrogram_H5_name)


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

        # write specufex parameters to the file
        specuattr = _overwrite_group_if_exists(fileLoad, "specufex_attrs")
        for attr in specparams.keys():
            specuattr.attrs[attr] = specparams[attr]
