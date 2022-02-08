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


def make_plot_Dist(cluster_name,df_merged_sub,var1,num_random = 100):
    plt.figure(figsize=(20,10))
    sns.violinplot(data=df_merged_sub, x=cluster_name, y=var1,palette="light:g",scale="width", inner="points", orient="v",cut=0)
    sns.stripplot(x=cluster_name, y=var1,
                    data=df_merged_sub,
                       size=4, color=".26")

    cluster_count = df_merged_sub.groupby(cluster_name)[cluster_name].count()
    cluster_indx = cluster_count.index.values
    cluster_num  = cluster_count.values
    value_sample = df_merged_sub[var1].values
    random_wasserstein_distance = np.zeros(num_random)
    random_energy_distance = np.zeros(num_random)
    for rand_count in range(0,num_random):
        value_sample_subsample =  value_sample.copy()
        subgroups = []
        for clust in range(0,cluster_indx.size) :
            #print(clust,cluster_num[clust])
            choice = np.random.choice(range(value_sample_subsample.shape[0]), size=(cluster_num[clust],), replace=False)
            ind = np.zeros(value_sample_subsample.shape[0], dtype=bool)
            ind[choice] = True
            rest = ~ind
            subgroups.append(value_sample_subsample[ind])
            value_sample_subsample = value_sample_subsample[rest]
            assert(subgroups[-1].shape[0] == cluster_num[clust])

        tmp = np.zeros([cluster_indx.size,cluster_indx.size])
        tmp2 = np.zeros([cluster_indx.size,cluster_indx.size])
        for i in range(0,cluster_indx.size) :
            for j in range(0,cluster_indx.size) :
                tmp[i,j] = wasserstein_distance(subgroups[i], subgroups[j])
                tmp2[i,j] = energy_distance(subgroups[i], subgroups[j])
        random_wasserstein_distance[rand_count] = np.mean(tmp[tmp>0])
        random_energy_distance[rand_count] = np.mean(tmp2[tmp2>0])

    cluster_wasserstein_distance = np.zeros([cluster_indx.size,cluster_indx.size])
    cluster_energy_distance = np.zeros([cluster_indx.size,cluster_indx.size])
    for i in range(0,cluster_indx.size) :
        xv =  df_merged_sub.loc[df_merged_sub[cluster_name]==cluster_indx[i],var1]
        for j in range(0,cluster_indx.size) :
            yv =  df_merged_sub.loc[df_merged_sub[cluster_name]==cluster_indx[j],var1]
            cluster_wasserstein_distance[i,j] = wasserstein_distance(xv, yv)
            cluster_energy_distance[i,j] = energy_distance(xv, yv)

    f,axs = plt.subplots(1,2,figsize=(20,10))
    sns.heatmap(cluster_wasserstein_distance,center=0, cmap="vlag",ax = axs[0])
    axs[0].set_title('Earth Mover Distance')
    mean_wasserstein_distance  = cluster_wasserstein_distance[cluster_wasserstein_distance>0].mean()

    sns.heatmap(cluster_energy_distance,center=0, cmap="vlag",ax = axs[1])
    axs[1].set_title('Energy Distance')
    mean_energy_distance  = cluster_energy_distance[cluster_energy_distance>0].mean()

    f,axs = plt.subplots(1,2,figsize=(20,10))
    sns.kdeplot(random_wasserstein_distance,color='blue',label=f'{num_random} Random Partitions (EM)',ax=axs[0])
    axs[0].axvline(mean_wasserstein_distance,color='k',label='Cluster Value (EM)')
    axs[0].legend()
    axs[0].set_xlabel('Wasserstein Distance Measure')

    sns.kdeplot(random_energy_distance,color='maroon',label=f'{num_random} Random Partitions (ED)',ax=axs[1])
    plt.axvline(mean_energy_distance,color='brown',label='Cluster Value (ED)')
    plt.xlabel('Energy Distance Measure')
    plt.legend()


def make_plot_Cross(cluster_name,df_merged_sub,var1,var2):
    plt.figure(figsize=(20,20))
    sns.displot(
        data=df_merged_sub, x=var1, y=var2, col=cluster_name,
        log_scale=(False, False), col_wrap=4, height=4, aspect=.7,
    )
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



# Get all the waveforms together :
# index_list = df_merged['Index_Exp'].values.astype(int).tolist()
def get_subset_waveforms_byexp(norm_waveforms,exp_ID,df_merged,select_subset=False,select_each=100,sample_rate=1,set_max_val = False,max_val_wave=1):
    index_list = []
    for exp in np.unique(exp_ID) :
        #print(exp)
        #indx = np.where(np.array(experiment_IDs) == exp)[0]
        indx = df_merged.loc[df_merged['Exp_Name'] == exp,'Index_Exp'].values
        if select_subset:
            index_list.append(np.random.choice(indx,select_each,replace=False))
        else :
            index_list.append(indx)

    index_list = np.array(index_list).flatten().tolist()
    waveforms_subset = []
    if set_max_val:
        for wave in [norm_waveforms[i] for i in index_list]:
            waveforms_subset.append(wave[:max_val_wave:sample_rate])
    else :
        for wave in [norm_waveforms[i] for i in index_list]:
            waveforms_subset.append(wave[::sample_rate])
    return index_list,waveforms_subset


# need to norm the correlations by (x^2*y^2)^0.5 like in obspy code
#numba.jit()
def corr_distance_1D(index_pair,matrix,num_shift_max,timeshifts):
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

def calc_corrmatrix(matrix,num_shift_max):
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

    master_index_list = []
    master_index_list = np.array(np.triu_indices(matlen)).T

    # for i in range(matlen):
    #     for j in range(i+1, matlen):
    #         master_index_list.append([i, j])
    t0 = time.time()
    num_cores = multiprocessing.cpu_count()-4
    print(f'Using {num_cores} cores')
    results = Parallel(n_jobs=num_cores)(delayed(corr_distance_1D)(i,matrix,num_shift_max,timeshifts) for i in tqdm(master_index_list))
    print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    for i_pair in range(len(results)):
        i = master_index_list[i_pair][0]
        j = master_index_list[i_pair][1]
        A[i, j] = results[i_pair][0]
        A[j, i] = results[i_pair][0]
        A_time[i, j] = results[i_pair][1]
        A_time[j, i] = -results[i_pair][1]
        A_summed[i, j] = results[i_pair][2]
        A_summed[j, i] = results[i_pair][2]
    return A,A_time,A_summed


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
            np.percentile(DTW_path_anomaly, 95),np.sum(np.abs(x[:,0] - f(time_vals))),path_linmdtw['path']]

def calc_corrmatrix_DTW(matrix,experiment_IDs,time_vals):
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

    t0 = time.time()
    num_cores = multiprocessing.cpu_count()-4
    print(f'Using {num_cores} cores')
    results = Parallel(n_jobs=num_cores)(delayed(corr_distance_1D_DTW)(i,matrix,time_vals) for i in tqdm(master_index_list))
    print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    for i_pair in range(len(results)):
        prefactor = 1
        i = master_index_list[i_pair][0]
        j = master_index_list[i_pair][1]
        exp_name_j = experiment_IDs[j]
        if exp_name_j.split("_")[-1] == 'pico':
            prefactor = 1#20./2. # since sampling for other data is once every 20 sec, while pico is once every 3*2 = 6 seconds
        DTW[i, j] = results[i_pair][0]*prefactor
        DTW[j, i] = results[i_pair][0]*prefactor

        DTW_corr[i, j] = results[i_pair][1]
        DTW_corr[j, i] = results[i_pair][1]

        DTW_L2[i, j] = results[i_pair][2]
        DTW_L2[j, i] = results[i_pair][2]

        DTW_time_med[i, j] = results[i_pair][3]*prefactor
        DTW_time_med[j, i] = -results[i_pair][3]*prefactor

        DTW_time_95[i, j] = results[i_pair][4]*prefactor
        DTW_time_95[j, i] = -results[i_pair][4]*prefactor

        DTW_L1[i, j] = results[i_pair][5]
        DTW_L1[j, i] = results[i_pair][5]

    return DTW,DTW_time_med,DTW_time_95,DTW_corr,DTW_L2,DTW_L1,results,master_index_list


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


def calc_distmatrix_Basic_Distances(matrix,metric,n_jobs=False):
    '''
    metrics to use :
    l1 (manhattan/cityblock)
    l2 (euclidean)
    cosine
    correlation
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    '''
    matrix = (matrix - np.mean(matrix, axis=1)[:,np.newaxis])

    t0 = time.time()
    if n_jobs :
        num_cores = multiprocessing.cpu_count()-4
    else :
        num_cores = 1
    print(f'Using {num_cores} cores')
    results = pairwise_distances(matrix,metric=metric,n_jobs=10)
    print(": {:.2f} s".format(time.time()-t0))
    gc.collect()
    return results




for exp_ID in np.unique(experiment_IDs):
    print(exp_ID)
    f1 = os.path.isfile(path_to_save+exp_ID+"/New_Processed_correlation_matrix.npy")
    time_vals = np.arange(0,num_events[exp_ID],resample_val)
    index_list,waveforms_subset = he.get_subset_waveforms_byexp(norm_waveforms,
                                                                    exp_ID,df_merged,set_max_val=True,select_subset=False,
                                                                 max_val_wave= num_events[exp_ID],sample_rate=resample_val)
    try :
            os.mkdir(path_to_save+exp_ID)
    except:
            print('Exists')
    if f1 == False :
        A,A_time,A_summed = he.calc_corrmatrix(waveforms_subset,int(num_events[exp_ID]*frac_allowed))
        np.save(path_to_save+exp_ID+"/New_Processed_correlation_matrix.npy", A)
        np.save(path_to_save+exp_ID+"/New_Processed_correlation_matrix_TimeShift.npy", A_time)
        np.save(path_to_save+exp_ID+"/New_Processed_correlation_matrix_Summed.npy", A_summed)

    f2 = os.path.isfile(path_to_save+exp_ID+"/Distance_Matrix_L1.npy")
    if f2 == False :
        A_L1 = he.calc_distmatrix_Basic_Distances(waveforms_subset,'l1',n_jobs=True)
        np.save(path_to_save+exp_ID+"/Distance_Matrix_L1.npy", A_L1)
        A_L2 = he.calc_distmatrix_Basic_Distances(waveforms_subset,'l2',n_jobs=True)
        np.save(path_to_save+exp_ID+"/Distance_Matrix_L2.npy", A_L2)
        A_cos = he.calc_distmatrix_Basic_Distances(waveforms_subset,'cosine',n_jobs=True)
        np.save(path_to_save+exp_ID+"/Distance_Matrix_cosine.npy", A_cos)
        A_cor = he.calc_distmatrix_Basic_Distances(waveforms_subset,'correlation',n_jobs=True)
        np.save(path_to_save+exp_ID+"/Distance_Matrix_correlation.npy", A_cor)
gc.collect()
