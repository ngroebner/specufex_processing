import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import resample, fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance,energy_distance
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances

import os
import time
import sys

import linmdtw
import matplotlib as mpl

from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

import seaborn as sns
import gc

plt.rcParams['axes.facecolor'] = 'white'

import warnings
warnings.filterwarnings("ignore")


def set_plot_prop():
    plt.ioff()
    mm2inch = lambda x: x/10./2.54
    # plt.rcParams['xtick.direction']= 'out'
    # plt.rcParams['ytick.direction']= 'out'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.75
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 2.
    plt.rcParams['figure.figsize'] = mm2inch(90*5),mm2inch(2./3*90*5)
    plt.rcParams["legend.handlelength"] = 1.
    plt.rcParams["legend.handletextpad"] = 0.15
    plt.rcParams["legend.borderpad"] = 0.15
    plt.rcParams["legend.labelspacing"] = 0.15
    cmap=plt.cm.get_cmap('RdYlBu')
    plt.rcParams.update({
        "figure.facecolor":  (1.0, 1.0, 1.0, 1),  # red   with alpha = 30%
        "axes.facecolor":    (1.0, 1.0, 1.0, 1),  # green with alpha = 50%
        "savefig.facecolor": (1.0, 1.0, 1.0, 1),  # blue  with alpha = 20%
    })

def plot_matrix(df_merged,evIDs_int,var1,var2,val,center,num_values_side_bar = 100):
    plt.figure(figsize=(20,10))
    row_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var1].astype('float')[evIDs_int]
    row_column = row_column - row_column.min()
    row_column = (row_column.values/np.nanmax(row_column.values))[:,np.newaxis]
    row_column = np.hstack([np.zeros((num_values_side_bar,num_values_side_bar)),
                            np.tile(row_column.transpose()*val.max(),(num_values_side_bar,1))])

    col_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var2].astype('float')[evIDs_int]
    col_column = col_column - col_column.min()
    col_column = (col_column.values/col_column.max())[:,np.newaxis]

    sns.heatmap(np.vstack([row_column,np.hstack([np.tile(col_column*val.max(),(1,num_values_side_bar)),val])]),center=center, cmap="vlag")

def plot_cluster(df_merged,evIDs_int,var1,var2,val,center,vmin=0,vmax=1000,vmin_use = False,method='ward', row_linkage=None,col_linkage=None):
    plt.figure(figsize=(20,10))

    row_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var2].astype('float')[evIDs_int]
    norm = mpl.colors.Normalize(vmin=row_column.min(), vmax=row_column.max(), clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.PuRd)
    row_colors = pd.DataFrame(row_column.values,columns=[var2])
    row_colors[var2] = row_colors[var2].apply(lambda x: mapper.to_rgba(x))

    col_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var1].astype('float')[evIDs_int]
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

    plt.figure(10,figsize=(6, 3))
    plt.colorbar(mapper2,cax=plt.gca(),orientation='horizontal', label=f'{var1} Column Colors')
    plt.gcf().subplots_adjust(bottom=0.45)
    plt.figure(11,figsize=(3, 6))
    plt.colorbar(mapper,cax=plt.gca(),orientation='vertical', label=f'{var2} Row Colors')
    plt.gcf().subplots_adjust(right=0.45)
    return g

def plot_cluster_Multi(df_merged,evIDs_int,var1_list,var2_list,val,center,vmin=0,vmax=1000,vmin_use = False,method='ward', row_linkage=None,col_linkage=None):
        var1 = var1_list[0]
        row_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var1].astype('float')[evIDs_int]
        norm = mpl.colors.Normalize(vmin=row_column.min(), vmax=row_column.max(), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdBu_r)
        row_colors = pd.DataFrame(row_column.values,columns=[var1])
        row_colors[var1] = row_colors[var1].apply(lambda x: mapper.to_rgba(x))

        for var1 in var1_list[1:] :
            print(var1)
            row_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var1].astype('float')[evIDs_int]
            norm = mpl.colors.Normalize(vmin=row_column.min(), vmax=row_column.max(), clip=True)
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdYlBu_r)
            row_colors_Tmp = pd.DataFrame(row_column.values,columns=[var1])
            row_colors[var1] = row_colors_Tmp[var1].apply(lambda x: mapper.to_rgba(x))

        var2 = var2_list[0]

        col_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var2].astype('float')[evIDs_int]
        norm = mpl.colors.Normalize(vmin=col_column.min(), vmax=col_column.max(), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdBu_r)
        col_colors = pd.DataFrame(col_column.values,columns=[var2])
        col_colors[var2] = col_colors[var2].apply(lambda x: mapper.to_rgba(x))

        for var2 in var2_list[1:] :
            print(var2)
            col_column = df_merged.loc[df_merged['Index_Exp'].isin(evIDs_int),var2].astype('float')[evIDs_int]
            norm = mpl.colors.Normalize(vmin=col_column.min(), vmax=col_column.max(), clip=True)
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdBu_r)
            col_colors_Tmp = pd.DataFrame(col_column.values,columns=[var2])
            col_colors[var2] = col_colors_Tmp[var2].apply(lambda x: mapper.to_rgba(x))


        if vmin_use :
            g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                                figsize=(12, 13), row_linkage=row_linkage,col_linkage=col_linkage,
                               vmin=vmin,vmax=vmax,
                               row_colors=row_colors,
                               col_colors=col_colors
                              )
        else :
            g = sns.clustermap(pd.DataFrame(val),center=center, cmap="GnBu",method = method,
                                figsize=(12, 13),
                               row_colors=row_colors,
                               col_colors=col_colors,
                               row_linkage=row_linkage,col_linkage=col_linkage,
                              )
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
    #plt.vlines([5471, 5471+12010, 5471+12010+900,5471+12010+900+1685], ymin=[0,0,0,0], ymax=[2,2,2,2])
