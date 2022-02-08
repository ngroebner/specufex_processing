# extract the fingerprints from the h5 SpecUFEx file,
# run k-means on them and merge with a catalog
import sys

import numpy as np
import h5py
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import pandas as pd

from importlib import reload
from specufex_processing.clustering import functions_clustering as funclust

import yaml
import argparse
import os

# command line argument instead of hard coding to config file
parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

with open(args.config_filename, 'r') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


# pull out config values for conciseness
path_config = config["paths"]
projectPath = path_config["projectPath"]
key = path_config["key"]

data_config = config['dataParams']
station = data_config["station"]
channel = data_config["channel"]
channel_ID = data_config["channel_ID"]
sampling_rate = data_config["sampling_rate"]


clustering_params = config['clusteringParams']
# build path strings
dataH5_name = f'data_{key}.h5'
dataH5_path = os.path.join(projectPath,'H5files/', dataH5_name)

SpecUFEx_H5_name = 'SpecUFEx_' + path_config["h5name"]
SpecUFEx_H5_path = os.path.join(projectPath, 'H5files/', SpecUFEx_H5_name)
pathWf_cat  = os.path.join(projectPath, 'wf_cat_out.csv')

sgram_catPath = os.path.join(projectPath,f'sgram_cat_out_{key}.csv')
cat0 = pd.read_csv(sgram_catPath)

path_cluster_base = os.path.join(config['paths']['projectPath'],"clustering_Catalog")
os.makedirs(path_cluster_base,exist_ok=True)

#### After linearizing FPs into data array X, you can then do Kmeans on X
## Optimal cluster is chosen by mean Silh scores, but euclidean distances are saved also
X,evID_hdf5,orig_df = funclust.linearizeFP(SpecUFEx_H5_path,cat0)

# ====================================================================
# Do the PCA (in case of clustering on PCA but also for visualization)
# ====================================================================

# # Clustering parameters
if clustering_params['use_PCA']:
    if clustering_params['use_PCA_method'] == 'num':
        PCA_df, Y_pca = funclust.PCAonFP(X,evID_hdf5,cat0,
                                            numPCA=clustering_params['numPCA'],
                                            normalization=True)
        X_use = Y_pca
        df_use = PCA_df
    elif clustering_params['use_PCA_method'] == 'var':
        PCA_df, Y_pca = funclust.PVEofPCA(X,evID_hdf5,cat0,
                                            cum_pve_thresh=clustering_params['cum_pve_thresh'],
                                            normalization=True)
        X_use = Y_pca
        df_use = PCA_df
    else :
        print(f"{clustering_params['use_PCA_method']} needs to num or var")
else :
        X_use = X
        df_use = orig_df

if clustering_params['runSilhouette'] == 'True':
    Kmax = clustering_params['Kmax']
    range_n_clusters = range(2,Kmax+1)

    # NOTE that i modified this so that you pass in X, be it fingerprints or PCA-- do that outside the function.
    df_use,catall_euc,catall_ss,Kopt, maxSilScore, avgSils,centers,sse = funclust.calcSilhScore(X_use,df_use,range_n_clusters,topF=clustering_params['topF'])
    catall_euc.to_csv(path_cluster_base+f"/Clustering_Kmeans_NC{Kopt}_Catalog_Sorted_By_Euc_Top{clustering_params['topF']}.csv")
    catall_ss.to_csv(path_cluster_base+f"/Clustering_Kmeans_NC{Kopt}_Catalog_Sorted_By_SS_Top{clustering_params['topF']}.csv")

else :
    ## use the passed values of K_vals
    sse = []
    centers = []
    for K_save in clustering_params['K_vals']:
        print(f"Running kmeans on {K_save} clusters, to save to catalog")
        kmeans = KMeans(n_clusters=K_save,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state
        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X_use)
        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]
        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X_use)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)
        #save centroids
        centers.append(kmeans.cluster_centers_ )
        #kmeans loss function
        sse.append(kmeans.inertia_)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        df_use[f'Cluster_NC{K_save}'] = cluster_labels
        df_use[f'SS_NC{K_save}'] = sample_silhouette_values
        df_use[f'euc_dist_NC{K_save}'] = euc_dist
        catall_euc,catall_ss = funclust.getTopF_labels(K_save,df_use,topF=clustering_params['topF'])
        catall_euc.to_csv(path_cluster_base+f"/Clustering_Kmeans_NC{Kopt}_Catalog_Sorted_By_Euc_Top{clustering_params['topF']}.csv")
        catall_ss.to_csv(path_cluster_base+f"/Clustering_Kmeans_NC{Kopt}_Catalog_Sorted_By_SS_Top{clustering_params['topF']}.csv")

df_use.to_csv(path_cluster_base+'/Clustering_Kmeans_Catalog.csv')

original_stdout = sys.stdout # Save a reference to the original standard output
with open(path_cluster_base+'/Clustering_Kmeans_Catalog_Info.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print('Parameters of the clustering Algorithm')
    print(clustering_params)
    print(f'Cluster Number {range_n_clusters}')
    print('kmeans loss function : ')
    print(sse)
    print('kmeans centroids : ')
    print(centers)
    sys.stdout = original_stdout # Reset the standard output to its original value
