import numpy as np

import os
import time
import sys

import linmdtw

from tqdm import tqdm
from joblib import Parallel, delayed

import gc



# #numba.jit()
# def corr_distance_1D(index_pair,matrix,plot=False):
#     i = index_pair[0]
#     j = index_pair[1]
#     x = matrix[i][:,np.newaxis]
#     y = matrix[j][:,np.newaxis]
#     path_linmdtw = linmdtw.linmdtw(x,y)
#     cost = linmdtw.get_path_cost(x, y, path_linmdtw)
#     #path_linmdtw = linmdtw.dtw_brute_backtrace(x,y, debug=True)
#     DTW_path_anomaly = (path_linmdtw[:, 1]-path_linmdtw[:, 0])

#     f = interp1d(time_vals[path_linmdtw[:,0]],y[path_linmdtw[:,1],0],kind='nearest',fill_value='extrapolate')
# #     if plot :
# #         plt.figure(figsize=(20,10))
# #         plt.plot(x,'blue')
# #         plt.plot(y,'green',alpha=0.5)
# #         plt.plot(time_vals[path_linmdtw['path'][:,0]],y[path_linmdtw['path'][:,1]],'r',alpha=1)
# #         plt.plot(time_vals,f(time_vals),'maroon',alpha=1)
#     return [np.sum(np.abs(DTW_path_anomaly))/(path_linmdtw[:,0].shape[0]),
#             np.corrcoef(x[:,0],f(time_vals))[0,1],
#             cost,
#             np.percentile(DTW_path_anomaly, 50),
#             np.percentile(DTW_path_anomaly, 95),cost,path_linmdtw]

def corr_distance_1D_DTW(index_pair,matrix,time_vals, plot=False):
    i = index_pair[0]
    j = index_pair[1]
    x = matrix[i][:,np.newaxis]
    y = matrix[j][:,np.newaxis]
    #path_linmdtw = linmdtw.linmdtw(x,y)
    #cost = linmdtw.get_path_cost(x, y, path_linmdtw)
    path_linmdtw = linmdtw.dtw_brute_backtrace(x,y, debug=True)
    DTW_path_anomaly = (path_linmdtw['path'][:, 1]-path_linmdtw['path'][:, 0])

    f = interp1d(time_vals[path_linmdtw['path'][:,0]],y[path_linmdtw['path'][:,1],0],kind='nearest',fill_value='extrapolate')
    if plot :
        plt.figure(figsize=(20,10))
        plt.plot(x,'blue')
        plt.plot(y,'green',alpha=0.5)
        plt.plot(time_vals[path_linmdtw['path'][:,0]],y[path_linmdtw['path'][:,1]],'r',alpha=1)
        plt.plot(time_vals,f(time_vals),'maroon',alpha=1)
    gc.collect()
    return [np.sum(np.abs(DTW_path_anomaly))/(path_linmdtw['path'][:,0].shape[0]),
            np.corrcoef(x[:,0],f(time_vals))[0,1],
            path_linmdtw['cost'],
            np.percentile(DTW_path_anomaly, 50),
            np.percentile(DTW_path_anomaly, 95),np.sum(np.abs(x[:,0] - f(time_vals)))] #,path_linmdtw['path']]

def calc_corrmatrix_DTW(matrix,time_vals,use_multi=True,num_cores=12):
    matlen = len(matrix)
    DTW = np.zeros((matlen,matlen))
    DTW_time_med = np.zeros((matlen,matlen))
    DTW_time_95 = np.zeros((matlen,matlen))
    DTW_corr = np.zeros((matlen,matlen))
    DTW_L2 = np.zeros((matlen,matlen))
    DTW_L1 = np.zeros((matlen,matlen))

    # demean the waveforms
    matrix = (matrix - np.mean(matrix, axis=1)[:,np.newaxis])

    master_index_list = []
    master_index_list = np.array(np.triu_indices(matlen)).T

    if use_multi :
        t0 = time.time()
        print(f'Using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(corr_distance_1D_DTW)(i,matrix,time_vals) for i in tqdm(master_index_list))
        print(": {:.2f} s".format(time.time()-t0))
        gc.collect()
        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            DTW[i, j] = results[i_pair][0]
            DTW[j, i] = results[i_pair][0]

            DTW_corr[i, j] = results[i_pair][1]
            DTW_corr[j, i] = results[i_pair][1]

            DTW_L2[i, j] = results[i_pair][2]
            DTW_L2[j, i] = results[i_pair][2]

            DTW_time_med[i, j] = results[i_pair][3]
            DTW_time_med[j, i] = -results[i_pair][3]

            DTW_time_95[i, j] = results[i_pair][4]
            DTW_time_95[j, i] = -results[i_pair][4]

            DTW_L1[i, j] = results[i_pair][5]
            DTW_L1[j, i] = results[i_pair][5]
    else :
        t0 = time.time()
        for index_pair in tqdm(master_index_list) :
            i = index_pair[0]
            j = index_pair[1]
            results = corr_distance_1D_DTW(index_pair,matrix,time_vals)

            DTW[i, j] = results[0]
            DTW[j, i] = results[0]

            DTW_corr[i, j] = results[1]
            DTW_corr[j, i] = results[1]

            DTW_L2[i, j] = results[2]
            DTW_L2[j, i] = results[2]

            DTW_time_med[i, j] = results[3]
            DTW_time_med[j, i] = -results[3]

            DTW_time_95[i, j] = results[4]
            DTW_time_95[j, i] = -results[4]

            DTW_L1[i, j] = results[5]
            DTW_L1[j, i] = results[5]
        print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    return DTW,DTW_time_med,DTW_time_95,DTW_corr,DTW_L2,DTW_L1
