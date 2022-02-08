import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import resample, fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance,energy_distance
from sklearn.metrics import pairwise_distances

import os
import time
import sys
import pdb

from tqdm import tqdm
from joblib import Parallel, delayed

import gc
import seaborn as sns

# import warnings
# warnings.filterwarnings("ignore")


# need to norm the correlations by (x^2*y^2)^0.5 like in obspy code
def corr_distance_1D(index_pair,matrix,num_shift_max,timeshifts):
    num_shift_max = int(num_shift_max)
    i = index_pair[0]
    j = index_pair[1]
    x = matrix[i]
    y = np.flipud(matrix[j]) #flipped_matrix[j]
    normconst = (np.power(x,2).sum()*np.power(y,2).sum())**0.5
    vals = fftconvolve(x,y,mode='full')/normconst
    cntr = int(vals.shape[0]/2)
    vals[:cntr-num_shift_max] = 0
    vals[cntr+num_shift_max:] = 0
    return [vals.max(),timeshifts[vals.argmax()],np.abs(vals).sum()]

def calc_corrmatrix(matrix,num_shift_max,use_multi=True,num_cores=12):
    timeshifts = np.arange(-matrix[0].shape[0]+1,matrix[0].shape[0])
    matlen = len(matrix)
    A = np.zeros((matlen,matlen))
    A_time = np.zeros((matlen,matlen))
    A_summed = np.zeros((matlen,matlen))
    # demean the waveforms
    matrix = (matrix - np.mean(matrix, axis=1)[:,np.newaxis])

    # calc ffts upfront
    #matrix_fft = fft(matrix)
    # calc ffts of reversed waveforms (correlation needs reversal of a sequence)
    #flipped_matrix_fft = fft(np.fliplr(matrix))
    #flipped_matrix = np.fliplr(matrix)

    # calc summed squares for each waveform for normalization
    # matrix_sumsq = np.power(matrix,2).sum(axis=1)

    master_index_list = np.array(np.triu_indices(matlen)).T

    # master_index_list = []
    # for i in range(matlen):
    #     for j in range(i+1, matlen):
    #         master_index_list.append([i, j])
    if use_multi :
        t0 = time.time()
        print(f'Using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(corr_distance_1D)(i,matrix,num_shift_max,timeshifts) for i in tqdm(master_index_list))
        print(": {:.2f} s".format(time.time()-t0))
        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            A[i, j] = results[i_pair][0]
            A[j, i] = results[i_pair][0]
            A_time[i, j] = results[i_pair][1]
            A_time[j, i] = -results[i_pair][1]
            A_summed[i, j] = results[i_pair][2]
            A_summed[j, i] = results[i_pair][2]
    else :
        t0 = time.time()
        for index_pair in tqdm(master_index_list) :
            i = index_pair[0]
            j = index_pair[1]
            results = corr_distance_1D(index_pair,matrix,num_shift_max,timeshifts)
            A[i, j] = results[0]
            A[j, i] = results[0]
            A_time[i, j] = results[1]
            A_time[j, i] = -results[1]
            A_summed[i, j] = results[2]
            A_summed[j, i] = results[2]
        print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    return A,A_time,A_summed

def corr_distance_2D(index_pair,matrix,num_shift_max,timeshifts):
    num_shift_max = int(num_shift_max)
    i = index_pair[0]
    j = index_pair[1]
    x = matrix[i]
    y = np.fliplr(matrix[j]) #flipped_matrix[j]
    normconst = (np.power(x,2).sum(axis=1)*np.power(y,2).sum(axis=1))**0.5
    vals = fftconvolve(x,y,mode='full',axes=1)/normconst[:,np.newaxis]
    return [vals.max(),timeshifts[vals.argmax(axis=1)].mean(),np.abs(vals).sum()]

def calc_corrmatrix_2D(matrix,num_shift_max,use_multi=True,num_cores=12):
    timeshifts = np.arange(-matrix[0].shape[-1]+1,matrix[0].shape[-1])
    matlen = len(matrix)
    A = np.zeros((matlen,matlen))
    A_time = np.zeros((matlen,matlen))
    A_summed = np.zeros((matlen,matlen))

    # calc ffts upfront
    #matrix_fft = fft(matrix)
    # calc ffts of reversed waveforms (correlation needs reversal of a sequence)
    #flipped_matrix_fft = fft(np.fliplr(matrix))
    #flipped_matrix = np.fliplr(matrix)

    # calc summed squares for each waveform for normalization
    # matrix_sumsq = np.power(matrix,2).sum(axis=1)

    master_index_list = np.array(np.triu_indices(matlen)).T

    # master_index_list = []
    # for i in range(matlen):
    #     for j in range(i+1, matlen):
    #         master_index_list.append([i, j])
    if use_multi :
        t0 = time.time()
        print(f'Using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(corr_distance_2D)(i,matrix,num_shift_max,timeshifts) for i in tqdm(master_index_list))
        print(": {:.2f} s".format(time.time()-t0))
        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            A[i, j] = results[i_pair][0]
            A[j, i] = results[i_pair][0]
            A_time[i, j] = results[i_pair][1]
            A_time[j, i] = -results[i_pair][1]
            A_summed[i, j] = results[i_pair][2]
            A_summed[j, i] = results[i_pair][2]
    else :
        t0 = time.time()
        for index_pair in tqdm(master_index_list) :
            i = index_pair[0]
            j = index_pair[1]
            results = corr_distance_2D(index_pair,matrix,num_shift_max,timeshifts)
            A[i, j] = results[0]
            A[j, i] = results[0]
            A_time[i, j] = results[1]
            A_time[j, i] = -results[1]
            A_summed[i, j] = results[2]
            A_summed[j, i] = results[2]
        print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    return A,A_time,A_summed


def calc_distmatrix_Basic_Distances(matrix,metric,use_multi=True,num_cores=12,use_2d = False):
    '''
    metrics to use :
    l1 (manhattan/cityblock)
    l2 (euclidean)
    cosine
    correlation
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    '''
    if use_2d == False :
        matrix = (matrix - np.mean(matrix, axis=1)[:,np.newaxis])
    else :
        matrix = matrix.reshape(matrix.shape[0],-1)

    t0 = time.time()
    if use_multi :
        n_cores = num_cores
    else :
        n_cores = 1
    print(f'Using {num_cores} cores')
    results = pairwise_distances(matrix,metric=metric,n_jobs=n_cores)
    print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    return results


def calculate_distances_all(X,distance_params,prefix_name,
                            distance_measure_name_list,
                            distance_measure_name_list_val,
                            use_cross_corr=True,use_2d=False,make_plots=True):
    '''
    Calculate a distance matrix from the input waveform grid
    '''
    plt.ioff()
    ########### Calculate the cross_correlation distance measures
    if use_cross_corr :
        distance_measure_name = '_Cross_Correlation'
        print(f'Calculating Distance Matrix for {distance_measure_name}')
        f1 = os.path.isfile(prefix_name+distance_measure_name+"_matrix.npy")
        if (f1 == False) | (distance_params['overwrite_distance'] == True) :
            if use_2d :
                A,A_time,A_summed = calc_corrmatrix_2D(X,distance_params['shift_allowed'],
                                                        use_multi=distance_params['multiprocessing'],
                                                        num_cores=distance_params['n_cores'])

            else :
                A,A_time,A_summed = calc_corrmatrix(X,distance_params['shift_allowed'],
                                                        use_multi=distance_params['multiprocessing'],
                                                        num_cores=distance_params['n_cores'])
            np.save(prefix_name+distance_measure_name+"_matrix.npy", A)
            np.save(prefix_name+distance_measure_name+"_matrix_TimeShift.npy", A_time)
            np.save(prefix_name+distance_measure_name+"_matrix_Summed.npy", A_summed)
            if make_plots:
                plt.figure(figsize=(20,10))
                sns.heatmap(A,center=np.median(A), cmap="vlag")
                plt.title(f'{distance_measure_name} Matrix')
                plt.savefig(prefix_name+distance_measure_name+"_matrix_Plot.png")
                plt.close()
                plt.figure(figsize=(20,10))
                sns.heatmap(A_time,center=np.median(A_time), cmap="vlag")
                plt.title(f'{distance_measure_name} Lag Time')
                plt.savefig(prefix_name+distance_measure_name+"_matrix_TimeShift_Plot.png")
                plt.close()
                plt.figure(figsize=(20,10))
                sns.heatmap(A_summed,center=np.median(A_summed), cmap="vlag")
                plt.title(f'{distance_measure_name} Summed')
                plt.savefig(prefix_name+distance_measure_name+"_matrix_Summed_Plot.png")
                plt.close()
            del A, A_time,A_summed
        else :
            print(f'Distance matrix for {distance_measure_name} {prefix_name} exists')
    ###############################################################
    ###############################################################
    ###############################################################
    for distance_measure_name,val in zip(distance_measure_name_list,distance_measure_name_list_val) :
        print(f'Calculating Distance Matrix for {distance_measure_name}')
        f1 = os.path.isfile(prefix_name+distance_measure_name+"_matrix.npy")
        if (f1 == False) | (distance_params['overwrite_distance'] == True) :
            A_L1 = calc_distmatrix_Basic_Distances(X,val,use_multi=distance_params['multiprocessing'],
                                                               num_cores=distance_params['n_cores'],use_2d = use_2d)
            np.save(prefix_name+distance_measure_name+"_matrix.npy", A_L1)
            if make_plots:
                plt.figure(figsize=(20,10))
                sns.heatmap(A_L1,center=np.median(A_L1), cmap="vlag")
                plt.title(f'{distance_measure_name}')
                plt.savefig(prefix_name+distance_measure_name+"_matrix_Plot.png")
                plt.close()
            del A_L1
        else :
            print(f'Distance matrix for {distance_measure_name} {prefix_name} exists')

    plt.ion()
    print("Done Calculations")
