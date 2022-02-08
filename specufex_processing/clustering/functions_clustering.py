import h5py
import pandas as pd
import numpy as np
#from obspy import read

import datetime as dtt

import datetime
from scipy.stats import kurtosis
from scipy import spatial

from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')
from scipy.io import loadmat
from sklearn.decomposition import PCA
# sys.path.append('.')
from sklearn.metrics import silhouette_samples
import scipy as sp
import scipy.signal

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler


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
