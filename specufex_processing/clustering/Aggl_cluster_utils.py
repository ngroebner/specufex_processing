import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score

from specufex_processing.plotting import Basic_config,Plotting_Utils

import os
import time
import sys

import matplotlib as mpl
from tqdm import tqdm
from joblib import Parallel, delayed

import seaborn as sns
import gc
import pdb

# import warnings
# warnings.filterwarnings("ignore")

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

def plot_cluster(df_merged,evIDs_int,var1,var2,val,center,vmin=0,
                        vmax=1000,vmin_use = False,method='ward',
                        row_linkage=None,col_linkage=None):
    plt.figure(figsize=(20,10))
    #pdb.set_trace()
    row_column = df_merged.loc[df_merged['ev_ID'].isin(evIDs_int),var2].astype('float')[evIDs_int]
    norm = mpl.colors.Normalize(vmin=row_column.min(), vmax=row_column.max(), clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.PuRd)
    row_colors = pd.DataFrame(row_column.values,columns=[var2])
    row_colors[var2] = row_colors[var2].apply(lambda x: mapper.to_rgba(x))

    col_column = df_merged.loc[df_merged['ev_ID'].isin(evIDs_int),var1].astype('float')[evIDs_int]
    norm2 = mpl.colors.Normalize(vmin=col_column.min(), vmax=col_column.max(), clip=True)
    mapper2 = mpl.cm.ScalarMappable(norm=norm2, cmap=mpl.cm.Greys)
    col_colors = pd.DataFrame(col_column.values,columns=[var1])
    col_colors[var1] = col_colors[var1].apply(lambda x: mapper2.to_rgba(x))

    if vmin_use :
        g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                            figsize=(12, 13), row_linkage=row_linkage,col_linkage=col_linkage,
                               row_colors=row_colors,
                               col_colors=col_colors,vmin=vmin,vmax=vmax,
                          )
    else :
        g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                            figsize=(12, 13), row_linkage=row_linkage,col_linkage=col_linkage,
                               row_colors=row_colors,
                               col_colors=col_colors,
                          )
    return g

def get_clust(df_merged_sub,n_clust,g,name_file,cluster_name):
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

    norm = mpl.colors.Normalize(vmin=1, vmax=n_clust, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Dark2)

    df_merged_sub[cluster_name] = clust
    df_merged_sub[cluster_name+'_plot_color'] = df_merged_sub[cluster_name].apply(lambda x: mapper.to_rgba(x))
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(name_file, 'w') as f:
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

def get_representative_waveforms(cluster_name,col_name,
                                norm_waveforms,
                                n_clust,
                                df_merged_sub,
                                index_list,save_path,
                                num_events=20,
                                time_shift_CrossCorr=None,
                                cross_corr=False,
                                start_clust=1):
    plt.ioff()
    if (cross_corr==True) :
        plt.figure(11,figsize=(20,10))
        plt.title('Shifted')
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

    for count in range(start_clust,n_clust+1):
        plt.figure(10)
        tmp = df_merged_sub.loc[df_merged_sub[cluster_name]==count,['ev_ID',col_name]]
        num_ev_ac = np.min([num_events,tmp.index.shape[0]])
        tmp2 = np.random.choice(tmp.ev_ID.values,num_ev_ac,replace=False)
        num_events_in_this = exp_num[exp_indx==count]
        my_yticks.append('Cluster '+str(count))
        #print(tmp2,num_events_in_this)

        if len(num_events_in_this) >0 :
            plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : {round(100*num_events_in_this[0]/exp_num.sum(),2)}', fontdict=font)
        else :
            plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : 0', fontdict=font)
        for k in range(0,num_ev_ac):
            indx_use = np.where(np.array(index_list) == tmp2[k])[0]
            x = norm_waveforms[indx_use].flatten()
            if k == 0:
                plt.plot(x + count*2,alpha=.2,label='Cluster : '+str(count),color=tmp[col_name].values[0])
            else :
                plt.plot(x + count*2,alpha=0.2,label='',color=tmp[col_name].values[0])
            #pdb.set_trace()

        if cross_corr :
                plt.figure(11)
                if len(num_events_in_this) >0 :
                    plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : {round(100*num_events_in_this[0]/exp_num.sum(),2)}', fontdict=font)
                else :
                    plt.text(wave_length*0.7, count*2+0.2, f'% of Waveforms : 0', fontdict=font)
                if num_ev_ac >0:
                    exemplar = np.where(np.array(index_list) == tmp2[0])[0]
                    x = norm_waveforms[exemplar].flatten()
                    plt.plot(x + count*2,alpha=0.2,label='Cluster : '+str(count),color=tmp[col_name].values[0])
                    for k in range(1,num_ev_ac):
                        indx_use = np.where(np.array(index_list) == tmp2[k])[0]
                        y = norm_waveforms[indx_use].copy().flatten()
                        shift = time_shift_CrossCorr[exemplar,indx_use][0].astype(int)
                        if shift > 0:
                            y[shift:] =  y[:-shift]
                        if shift < 0:
                             y[:shift] = y[-shift:]
                        plt.plot(y + count*2,alpha=0.2,label='',color=tmp[col_name].values[0])

    if (cross_corr==True):
        plt.figure(11)
        # leg = plt.legend(loc='best')
        # for line in leg.get_lines():
        #     line.set_linewidth(2.0)
        #     line.set_alpha(1)
        plt.xlim(left=0,right=wave_length)
        plt.xlabel('Waveform Time')
        plt.yticks(np.arange(start_clust*2,n_clust*2+1,2), my_yticks)
        plt.savefig(save_path+'_Shifted.png')
        plt.close()

    plt.figure(10)
    plt.xlim(left=0,right=wave_length)
#     leg = plt.legend(loc='best')
#     for line in leg.get_lines():
#             line.set_linewidth(2.0)
#             line.set_alpha(1)
    #print(my_yticks,np.arange(0,n_clust*2,2))
    plt.xlabel('Waveform Time')
    plt.yticks(np.arange(start_clust*2,n_clust*2+1,2), my_yticks)
    plt.savefig(save_path+'_Original.png')
    plt.close()
    plt.ion()
    #print('here')
