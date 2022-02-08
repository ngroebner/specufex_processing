import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score

import os
import time
import sys

import matplotlib as mpl
from tqdm import tqdm
from joblib import Parallel, delayed

import seaborn as sns
import gc

plt.rcParams['axes.facecolor'] = 'white'

import warnings
warnings.filterwarnings("ignore")


def plot_cluster(val,center,vmin=0,vmax=1000,vmin_use = False,
                 method='ward', row_linkage=None,col_linkage=None):
    plt.ioff()
    plt.figure(figsize=(20,10))

    if vmin_use :
        g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                            figsize=(12, 13), row_linkage=row_linkage,col_linkage=col_linkage,
                            vmin=vmin,vmax=vmax,
                          )
    else :
        g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                            figsize=(12, 13), row_linkage=row_linkage,col_linkage=col_linkage,
                          )
    plt.ion()
    return g

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def get_clust(df_merged_sub,n_clust,g,name,cluster_name):
    Z = g.dendrogram_col.linkage
    clust = fcluster(Z, n_clust, criterion="maxclust")

    plt.figure(figsize=(20,10))


    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=n_clust,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    #den = dendrogram(Z, p=n_clust, truncate_mode="lastp", labels = df_merged.index)

    #plt.show()
    df_merged_sub[cluster_name] = clust
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(name, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(f"Cluster Names : {np.unique(df_merged_sub[cluster_name])}")
        print(df_merged_sub.groupby(cluster_name)[cluster_name].count())
        sys.stdout = original_stdout # Reset the standard output to its original value

    return df_merged_sub

def Plot_clust_time(cluster_name,df_merged_sub):
    plt.figure(figsize=(20,2))
    plt.scatter(range(len(df_merged_sub[cluster_name])), np.ones_like(df_merged_sub[cluster_name]),
                c=df_merged_sub[cluster_name], s=1000, marker="|",cmap=plt.cm.rainbow)
    plt.colorbar(aspect=1)
    plt.tight_layout()

def get_representative_waveforms(cluster_name,norm_waveforms,time_vals,n_clust,df_merged_sub,index_list,num_events=20,is_DTW=False,time_shift_CrossCorr=None,cross_corr=False,max_time_indx=20000,max_val_time_wave=-1,resample_rate=1,start_clust=1,exp_name=''):
    if (cross_corr==True) | (is_DTW == True) :
        plt.figure(11,figsize=(20,10))
        plt.title('Shifted')
    plt.figure(10,figsize=(20,10))
    plt.title(f'Experiment Name : {exp_name}')
    my_yticks = []
    exp_count = df_merged_sub.groupby(cluster_name)[cluster_name].count()
    exp_indx = exp_count.index.values
    exp_num  = exp_count.values

    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }

    for count in range(start_clust,n_clust+1):
        plt.figure(10)
        tmp = df_merged_sub.loc[df_merged_sub[cluster_name]==count,'Index_Exp']
        #print(tmp.values,count)
        num_ev_ac = np.min([num_events,tmp.index.shape[0]])
        tmp2 = np.random.choice(tmp.values,num_ev_ac,replace=False)
        #print(tmp2,tmp.index)
        num_events_in_this = exp_num[exp_indx==count]
        my_yticks.append('Cluster '+str(count))

        if len(num_events_in_this) >0 :
            plt.text(700, count*2+0.2, f'% of Waveforms : {round(100*num_events_in_this[0]/exp_num.sum(),2)}', fontdict=font)
        else :
            plt.text(700, count*2+0.2, f'% of Waveforms : 0', fontdict=font)
        for k in range(0,num_ev_ac):
            x = norm_waveforms[tmp2[k]]
            if k == 0:
                plt.plot(time_vals,x[:max_val_time_wave*resample_rate:resample_rate] + count + count,alpha=0.1,label='Cluster : '+str(count),color=plt.cm.Dark2(count))
            else :
                plt.plot(time_vals,x[:max_val_time_wave*resample_rate:resample_rate] + count + count,alpha=0.1,label='',color=plt.cm.Dark2(count))
        if cross_corr :
                plt.figure(11)
                if num_ev_ac >0:
                    x = norm_waveforms[tmp2[0]]
                    plt.plot(time_vals,x[:max_val_time_wave*resample_rate:resample_rate] + count*2,alpha=0.1,label='Cluster : '+str(count),color=plt.cm.Dark2(count))
                    exemplar = np.where(index_list == tmp2[0])[0]
                    for k in range(1,num_ev_ac):
                        y = norm_waveforms[tmp2[k]].copy()
                        y = y[:max_val_time_wave*resample_rate:resample_rate]
                        shift = time_shift_CrossCorr[exemplar,np.where(index_list == tmp2[k])[0]][0].astype(int)
                        #print(shift,count,k)
                        if shift > 0:
                            y[shift:] =  y[:-shift]
                        if shift < 0:
                             y[:shift] = y[-shift:]
                        plt.plot(time_vals,y + count*2,alpha=0.1,label='',color=plt.cm.Dark2(count))
        if is_DTW :
            ################# Plotting the warped waveforms .. For DTW part ..
                plt.figure(11)
                if num_ev_ac >0:
                    x = norm_waveforms[tmp2[0]][:max_val_time_wave*resample_rate:resample_rate,np.newaxis]
                    #print(x.shape,time_vals.shape)
                    plt.plot(time_vals,x[:,0] + count*2,alpha=0.1,label='Cluster : '+str(count),color=plt.cm.Dark2(count))
                    exemplar = tmp2[0]
                    for k in range(1,num_ev_ac):
                        y =  norm_waveforms[tmp2[k]][:max_val_time_wave*resample_rate:resample_rate,np.newaxis]
                        path_linmdtw = linmdtw.dtw_brute_backtrace(x,y, debug=True)
                        f = interp1d(time_vals[path_linmdtw['path'][:,0]],y[path_linmdtw['path'][:,1],0],kind='nearest',fill_value='extrapolate')
                        #plt.plot(time_vals[path_linmdtw['path'][:,0]],y[path_linmdtw['path'][:,1]]+ count,alpha=0.25,label='',color=plt.cm.tab20(count))
                        plt.plot(time_vals,f(time_vals)+ count*2,alpha=0.1,label='',color=plt.cm.Dark2(count))

    if (cross_corr==True) | (is_DTW == True) :
        plt.figure(11,figsize=(20,10))
        leg = plt.legend(loc='best')
        for line in leg.get_lines():
            line.set_linewidth(2.0)
            line.set_alpha(1)
        plt.xlim(right=max_time_indx,left=0)

    plt.figure(10,figsize=(20,10))
#     leg = plt.legend(loc='best')
#     for line in leg.get_lines():
#             line.set_linewidth(2.0)
#             line.set_alpha(1)
    plt.xlim(right=max_time_indx,left=0)
    #print(my_yticks,np.arange(0,n_clust*2,2))
    plt.xlabel('Waveform Time')
    plt.yticks(np.arange(start_clust*2,n_clust*2+1,2), my_yticks)


def get_silloute_scores(A_L1,cluster_labels,n_clust):
    silhouette_avg = silhouette_score(A_L1, cluster_labels,metric='precomputed')
    print(
            "For n_clusters =",
            n_clust,
            "The average silhouette_score is :",
            silhouette_avg,
        )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(A_L1, cluster_labels,metric='precomputed')

    y_lower = 10
    plt.figure(figsize=(20,10))
    for i in range(n_clust):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

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
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
