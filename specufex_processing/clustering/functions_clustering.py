import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import os
import time
import sys

#from obspy import read

#import datetime as dtt
#import datetime
#from scipy.stats import kurtosis
#from scipy import spatial
#from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')
#from scipy.io import loadmat
# import scipy as sp
# import scipy.signal

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
import matplotlib as mpl

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import seaborn as sns
import gc


##################################################################################################
# ##################################################################################################
#        _           _            _
#       | |         | |          (_)
#    ___| |_   _ ___| |_ ___ _ __ _ _ __   __ _
#   / __| | | | / __| __/ _ \ '__| | '_ \ / _` |
#  | (__| | |_| \__ \ ||  __/ |  | | | | | (_| |
#   \___|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
#                                          __/ |
#                                         |___/
##################################################################################################

def linearizeFP(SpecUFEx_H5_path,cat0):
    X = []
    evID_hdf5 = []
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        for evID in MLout['fingerprints']:
            fp = MLout['fingerprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)
            evID_hdf5.append(evID)

    X = np.array(X)
    cols = [f'X{pp}' for pp in range(1,X.shape[1]+1)]
    pca_df = pd.DataFrame(data=X,columns=cols)
    pca_df['ev_ID'] = evID_hdf5
    cat0['ev_ID'] = cat0['ev_ID'].astype(str)
    PCA_df = cat0.merge(pca_df, right_on='ev_ID', left_on='ev_ID')
    return X,evID_hdf5,PCA_df

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PCAonFP(X,evID_hdf5,cat0,numPCA=3,normalization=None):
    ## performcs pca on fingerprints, returns catalog with PCs for each event
    if normalization=='RobustScaler':
        X_st = RobustScaler().fit_transform(X)
    if normalization=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif normalization=='MinMax':
        X_st = MinMaxScaler((-1,1)).fit_transform(X)
    else:
        X_st = X


    sklearn_pca = PCA(n_components=numPCA)
    Y_pca = sklearn_pca.fit_transform(X_st)
    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]
    pca_df = pd.DataFrame(data=Y_pca,columns=pc_cols)
    pca_df['ev_ID'] = evID_hdf5
    cat0['ev_ID'] = cat0['ev_ID'].astype(str)
    PCA_df = cat0.merge(pca_df, right_on='ev_ID', left_on='ev_ID')
    return PCA_df, Y_pca

def PVEofPCA(X,evID_hdf5,cat0,cum_pve_thresh=.8,normalization=None):
    ## performcs pca on fingerprints, returns catalog with PCs for each event
    if normalization=='RobustScaler':
        X_st = RobustScaler().fit_transform(X)
    if normalization=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif normalization=='MinMax':
        X_st = MinMaxScaler((-1,1)).fit_transform(X)
    else:
        X_st = X

    numPCMax=int(X.shape[1]) - 10
    numPCA_range = range(1,numPCMax)
    for numPCA in numPCA_range:
        sklearn_pca = PCA(n_components=numPCA)
        Y_pca = sklearn_pca.fit_transform(X_st)
        pve = sklearn_pca.explained_variance_ratio_
        cum_pve = pve.sum()
        if cum_pve >= cum_pve_thresh:
            break
    print(f'Using {numPCA} PCA components with {cum_pve} variance explained')
    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]
    pca_df = pd.DataFrame(data=Y_pca,columns=pc_cols)
    pca_df['ev_ID'] = evID_hdf5
    cat0['ev_ID'] = cat0['ev_ID'].astype(str)
    PCA_df = cat0.merge(pca_df, right_on='ev_ID', left_on='ev_ID')
    return PCA_df, Y_pca

def calcSilhScore(X,cat0,range_n_clusters,topF=5):

    """
    Parameters
    ----------
    range_n_clusters : range type - 2 : Kmax clusters

    Returns
    -------
    Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters.

    """

    ## Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters
    ## Returns altered cat00 dataframe with cluster labels and SS scores,

    maxSilScore = 0
    distMeasure = "SilhScore"
    sse = []
    avgSils = []
    centers = []

    for n_clusters in range_n_clusters:
        print(f"kmeans on {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X)
        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)
        #save centroids
        centers.append(kmeans.cluster_centers_ )
        #kmeans loss function
        sse.append(kmeans.inertia_)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        #%  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)
        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            euc_dist_best = euc_dist
            ss_best       = sample_silhouette_values

    print(f"Best cluster: {Kopt}")
    cat0[f'Cluster_NC{Kopt}'] = cluster_labels_best
    cat0[f'SS_NC{Kopt}'] = ss_best
    cat0[f'euc_dist_NC{Kopt}'] = euc_dist_best

    norm = mpl.colors.Normalize(vmin=1, vmax=Kopt, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Dark2)
    cat0[f'plot_color_NC{Kopt}'] = cat0[f'Cluster_NC{Kopt}'].apply(lambda x: mapper.to_rgba(x))

    catall_euc,catall_ss = getTopF_labels(Kopt,cat0,topF=topF)
    return cat0,catall_euc,catall_ss,Kopt, maxSilScore, avgSils,centers,sse

def getTopF_labels(Kopt,cat0,topF=1):
    ## make df for  top SS score rep events
    catall_ss = pd.DataFrame();
    catall_euc = pd.DataFrame();

    for k in range(1,Kopt+1):
        tmp_top0 = cat0.where(cat0[f'Cluster_NC{Kopt}']==k).dropna();
        tmp_top0_ss = tmp_top0.sort_values(by=f'SS_NC{Kopt}',ascending=False)
        tmp_top0_ss = tmp_top0_ss[0:topF]
        catall_ss = catall_ss.append(tmp_top0_ss);

        tmp_top0_eu = tmp_top0.sort_values(by=f'euc_dist_NC{Kopt}',ascending=False)
        tmp_top0_eu = tmp_top0_eu[0:topF]
        catall_euc = catall_euc.append(tmp_top0_eu);
    return catall_euc,catall_ss

def plot_silloute_scores_kmeans(sample_silhouette_values,cluster_labels,n_clust,path_cluster_plot_save):
    # Compute the silhouette scores for each sample
    y_lower = 10
    plt.ioff()
    plt.figure(figsize=(20,10))
    for i in range(1,n_clust+1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[np.where(np.array(cluster_labels) == i)[0]]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        #print(y_upper,y_lower,ith_cluster_silhouette_values)
        color = plt.cm.nipy_spectral(float(i) / n_clust)
        plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1 = plt.gca()
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 1  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    plt.savefig(path_cluster_plot_save)
    plt.close()
    plt.ion()

def get_representative_waveforms_kmeans(cluster_name,col_name,
                                        norm_waveforms,
                                        n_clust,df_merged_sub,best_rep,
                                        index_list,save_path,
                                        num_events=20,
                                        start_clust=1):
    plt.ioff()
    plt.figure(10,figsize=(20,10))
    my_yticks = []
    exp_count = df_merged_sub.groupby(cluster_name)[cluster_name].count()
    exp_indx = exp_count.index.values
    exp_num  = exp_count.values
    wave_length = norm_waveforms[0].shape[0]
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
    #print(exp_count)
    for count in range(start_clust,n_clust+1):
        tmp = df_merged_sub.loc[df_merged_sub[cluster_name]==count,['ev_ID',col_name]]
        tmp_Primary = best_rep.loc[best_rep[cluster_name]==count,['ev_ID',col_name]]
        #print(tmp.values,count)
        num_ev_ac = np.min([num_events,tmp.index.shape[0]])
        tmp2 = np.random.choice(tmp.ev_ID.values,num_ev_ac-1,replace=False)
        tmp2 = np.hstack([tmp_Primary.ev_ID.values,tmp2])
        num_events_in_this = exp_num[exp_indx==count]
        my_yticks.append('Cluster '+str(count))

        if len(num_events_in_this) >0 :
            plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : {round(100*num_events_in_this[0]/exp_num.sum(),2)}', fontdict=font)
        else :
            plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : 0', fontdict=font)
        for k in range(0,num_ev_ac):
            indx_use = np.where(np.array(index_list) == tmp2[k])[0]
            x = norm_waveforms[indx_use]
            #print(num_ev_ac,tmp2[k])
            if k == 0:
                plt.plot(x.flatten() + count + count,alpha=.2,label='Cluster : '+str(count),color=tmp[col_name].values[0])
            else :
                plt.plot(x.flatten() + count + count,alpha=0.05,label='',color=tmp[col_name].values[0])

    plt.xlabel('Waveform Time')
    plt.xlim(left=0,right=wave_length)
    plt.yticks(np.arange(start_clust*2,n_clust*2+1,2), my_yticks)
    plt.savefig(save_path)
    plt.close()
    plt.ion()
