

import h5py
import pandas as pd
import numpy as np
from obspy import read

import datetime as dtt

import datetime
from scipy.stats import kurtosis
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from scipy import spatial

from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')

from scipy.io import loadmat

# sys.path.append('.')
import scipy as sp
import scipy.signal




##########################################################################################


def butter_bandpass(fmin, fmax, fs, order=5):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fmin, fmax, fs, order=5):
    b, a = butter_bandpass(fmin, fmax, fs, order=order)
    y = lfilter(b, a, data)
    return y

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

##################################################################################################
##################################################################################################
#   _               _            _       _                          _                 _   _
#  | |             (_)          | |     | |                        | |               | | (_)
#  | |__   __ _ ___ _  ___    __| | __ _| |_ __ _    _____  ___ __ | | ___  _ __ __ _| |_ _  ___  _ __
#  | '_ \ / _` / __| |/ __|  / _` |/ _` | __/ _` |  / _ \ \/ / '_ \| |/ _ \| '__/ _` | __| |/ _ \| '_ \
#  | |_) | (_| \__ \ | (__  | (_| | (_| | || (_| | |  __/>  <| |_) | | (_) | | | (_| | |_| | (_) | | | |
#  |_.__/ \__,_|___/_|\___|  \__,_|\__,_|\__\__,_|  \___/_/\_\ .__/|_|\___/|_|  \__,_|\__|_|\___/|_| |_|
#                                                            | |
#                                                            |_|
##################################################################################################



def getWF(evID,dataH5_path,station,channel,fmin,fmax,fs):

    with h5py.File(dataH5_path,'a') as fileLoad:

        wf_data = fileLoad[f'waveforms/{station}/{channel}'].get(str(evID))[:]

    wf_filter = butter_bandpass_filter(wf_data, fmin,fmax,fs,order=4)
    wf_zeromean = wf_filter - np.mean(wf_filter)

    return wf_zeromean



def getSpectra(evID,station,path_proj,normed=True):

    if normed == False:
        ##right now saving normed
        try:
            mat = loadmat(f'{path_proj}01_input/{station}/specMats_nonNormed/{evID}.mat')
        except:
            mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')

    else:

        mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')


    specMat = mat.get('STFT')

    matSum = specMat.sum(1)

    return matSum,specMat

def getSpectra_fromWF(evID,dataH5_path,station,channel,normed=True):
## get WF from H5 and calc full sgram for plotting

    with h5py.File(dataH5_path,'r') as dataFile:

        wf_data = dataFile[f'waveforms/{station}/{channel}'].get(str(evID))[:]


        fs = dataFile['spec_parameters/'].get('fs')[()]

        # fmin =
        nperseg = dataFile['spec_parameters/'].get('nperseg')[()]
        noverlap = dataFile['spec_parameters/'].get('noverlap')[()]
        nfft = dataFile['spec_parameters/'].get('nfft')[()]


        fmax = dataFile['spec_parameters/'].get('fmax')[()]
        fmax = np.ceil(fmax)
        fmin = dataFile['spec_parameters/'].get('fmin')[()]
        fmin = np.floor(fmin)
        fSTFT = dataFile['spec_parameters/'].get('fSTFT')[()]
        tSTFT = dataFile['spec_parameters/'].get('tSTFT')[()]

        sgram_mode = dataFile['spec_parameters/'].get('mode')[()].decode('utf-8')
        scaling = dataFile['spec_parameters/'].get('scaling')[()].decode('utf-8')


    fs = int(np.ceil(fs))

    fSTFT, tSTFT, STFT_0 = sp.signal.spectrogram(x=wf_data,
                                                fs=fs,
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                #nfft=Length of the FFT used, if a zero padded FFT is desired
                                                nfft=nfft,
                                                scaling=scaling,
                                                axis=-1,
                                                mode=sgram_mode)

    if normed:
        STFT_norm = STFT_0 / np.median(STFT_0)  ##norm by median
    else:
        STFT_norm = STFT_0
    STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0)  ##convert to dB
    specMat = np.maximum(0, STFT_dB) #make sure nonnegative
    specMatsum = specMat.sum(1)

    return specMatsum,specMat,fSTFT

def getSpectraMedian(path_proj,cat00,k,station,normed=True):
    catk = cat00[cat00.Cluster == k]

    for j,evID in enumerate(catk.event_ID.iloc):

        if normed==False:
            matSum,specMat = getSpectra(evID,station,path_proj,normed=False)

        else:
            matSum,specMat = getSpectra(evID,station,path_proj,normed=True)

        if j == 0:
            specMatsum_med = np.zeros(len(matSum))


        specMatsum_med = np.vstack([specMatsum_med,matSum])



    specMatsum_med = np.median(specMatsum_med,axis=0)
    return specMatsum_med


def getSgram(path_proj,evID,station,tSTFT=[0]):


    mat = loadmat(f'{path_proj}01_input/{station}/specMats/{evID}.mat')

    specMat = mat.get('STFT')
    date = pd.to_datetime('200' + str(evID))

    x = [date + dtt.timedelta(seconds=i) for i in tSTFT]

    return specMat,x



def makeHourlyDF(ev_perhour_clus):

    """
    Returns dataframe of events binned by hour of day

    ev_perhour_resamp : pandas dataframe indexed by datetime

    """
    ev_perhour_resamp = ev_perhour_clus.resample('H').event_ID.count()



    hour_labels = list(ev_perhour_resamp.index.hour.unique())

    hour_labels.sort()
    #

    ev_perhour_resamp_list = list(np.zeros(len(hour_labels)))
    ev_perhour_mean_list = list(np.zeros(len(hour_labels)))




    hour_index = 0

    for ho in range(len(hour_labels)):
        hour_name = hour_labels[hour_index]
        ev_count = 0

#         print(hour_name)
        for ev in range(len(ev_perhour_resamp)):

            if ev_perhour_resamp.index[ev].hour == hour_name:
                ev_perhour_resamp_list[ho] += ev_perhour_resamp[ev]

                ev_count += 1

#         print(str(ev_count) + ' events in hour #' + str(hour_name))

        ev_perhour_mean_list[ho] = ev_perhour_resamp_list[ho] / ev_count

        hour_index += 1



##TS 2021/06/17 -- TS adjust hours here to CET
    hour_labels = [h + 2 for h in hour_labels]
    hour_labels[hour_labels==24] = 0
    hour_labels[hour_labels==25] = 1

    ev_perhour_resamp_df = pd.DataFrame({ 'EvPerHour' : ev_perhour_resamp_list,
                                          'MeanEvPerHour' : ev_perhour_mean_list},
                          index=hour_labels)


    ev_perhour_resamp_df['Hour'] = hour_labels



    return ev_perhour_resamp_df


def getDailyTempDiff(garciaDF_H,garciaDF_D,**plt_kwargs):

    tstart      =     plt_kwargs['tstartreal']
    tend        =     plt_kwargs['tendreal']

    garciaDF_H1 = garciaDF_H[garciaDF_H.datetime>=tstart]
    garciaDF_H1 = garciaDF_H1[garciaDF_H1.datetime<tend]

    garciaDF_D1 = garciaDF_D[garciaDF_D.datetime>=tstart]
    garciaDF_D1 = garciaDF_D1[garciaDF_D1.datetime<tend]


    temp_H = garciaDF_H1.tempC
    temp_H_a = np.array(temp_H)

    temp_H_a_r = temp_H_a.reshape(len(garciaDF_D1),24)
    mean_diff = []
    for i in range(len(temp_H_a_r[:,0])):
    #     plt.plot(temp_H_a_r[i,:] - garciaDF_D1.temp_D.iloc[i])
        mean_diff.append(temp_H_a_r[i,:] - garciaDF_D1.temp_D.iloc[i])


    mean_mean_diff = np.mean(mean_diff,axis=0)
    return mean_mean_diff
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getFeatures(catalog,filetype,fmin,fmax,fs,path_WF,nfft,dataH5_path,station,channel):

    columns=['event_ID','datetime','datetime_index','Cluster','RSAM','SC','P2P','VAR']
    df = pd.DataFrame(columns=columns)
    # RSAM_norm = 0
    # P2P_norm = 0
    # SC_norm = 0
    # VAR_norm = 0
    # byCluster=1

    for i,evID in enumerate(catalog.event_ID):



        wf_filter = getWF(evID,dataH5_path,station,channel,fmin,fmax,fs)

        date = pd.to_datetime(catalog.datetime.iloc[i])
        cluster = catalog.Cluster.iloc[i]


        RSAM = np.log10(np.sum(np.abs(wf_filter)))
#         RSAM_norm = RSAM_norm + (RSAM / np.max(RSAM))
        sc = np.mean(librosa.feature.spectral_centroid(y=np.array(wf_filter), sr=fs))


        # f = np.fft.fft(wf_filter)
        # f_real = np.real(f)
        # mag_spec = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=len(f)+nfft*2)[0]
        # freqs = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=len(f)+nfft*2)[1]
        # dominant_freq = freqs[np.where(mag_spec == mag_spec.max())]




#         SC_norm = SC_norm + (sc / np.max(sc))
        p2p = np.log10(np.max(wf_filter) - np.min(wf_filter))
#         P2P_norm = P2P_norm + (p2p / np.max(p2p))
        var = np.var(wf_filter)
#         VAR_norm = VAR_norm + (var / np.max(var))
        kurt = kurtosis(wf_filter)

        df = df.append(
                  {'event_ID':evID,
                   'datetime':date,
                   'datetime_index':date,
                   'Cluster':cluster,
                   'log10RSAM':RSAM,
                   'SC':sc,
                   'log10P2P':p2p,
                   'VAR':var,
                   'kurtosis':kurt},
                   ignore_index=True)


    df['RSAM_norm'] = [r/df.RSAM.max() for r in df.RSAM]
    df['SC_norm'] = [r/df.SC.max() for r in df.SC]
    df['P2P_norm'] = [r/df.P2P.max() for r in df.P2P]
    df['VAR_norm'] = [r/df.RSAM.max() for r in df.VAR]

    df = df.set_index('datetime_index')

    return df




def getLocationFeatures(map_catalog,stn,station):

    columns=['event_ID','datetime','datetime_index','Cluster','Elevation_m','Depth_m','DistXY_m','DistXYZ_m']
    df_loc = pd.DataFrame(columns=columns)

    stnX = np.array(stn[stn.name=='G7'+station].X)[0]
    stnY = np.array(stn[stn.name=='G7'+station].Y)[0]

    for i,evID in enumerate(map_catalog.event_ID):

        XX = map_catalog.X_m.iloc[i]
        YY = map_catalog.Y_m.iloc[i]
        elev = map_catalog.Elevation_m.iloc[i]
        ZZ = map_catalog.Depth_m.iloc[i]

        # XXabs = np.abs(XX)
        # YYabs = np.abs(YY)
        # ZZabs = ZZ

        disstXY = spatial.distance.euclidean([XX,YY],[stnX,stnY])
        disstXYZ = spatial.distance.euclidean([XX,YY,ZZ],[stnX,stnY,0])


#         X.loc[index, f'distXY_m'] = disstXY
#         X.loc[index, f'depth_m'] = ZZ


        date = pd.to_datetime(map_catalog.datetime.iloc[i])
        cluster = map_catalog.Cluster.iloc[i]

        df_loc = df_loc.append(
                  {'event_ID':evID,
                   'datetime':date,
                   'datetime_index':date,
                   'Cluster':cluster,
                   'Elevation_m':elev,
                   'Depth_m':ZZ,
                    'DistXY_m':disstXY,
                   'DistXYZ_m':disstXYZ},
                    ignore_index=True)

    df_loc = df_loc.set_index('datetime_index')

    return df_loc




def getNMFOrder(W,numPatterns):
    maxColVal = np.zeros(numPatterns)
    maxColFreq = np.zeros(numPatterns)

    order = list(range(0,numPatterns))

    for j in range(len(W[1])):
        maxColFreq[j] = W[:,j].argmax()
        maxColVal[j] = W[:,j].max()

    #% make dict of rearranged NMF dict


    W_df = pd.DataFrame({'order':order,
                          'maxColFreq':maxColFreq,
                          'maxColVal':maxColVal
                          })

    W_df_sort = W_df.sort_values(by='maxColFreq')


    order_swap = list(W_df_sort.order)

    return order_swap


def resortByNMF(matrix,order_swap):

    matrix_new = matrix.copy()

    for o in range(matrix.shape[1]):

        o_swap = order_swap[o]

        matrix_new[:,o] = matrix[:,o_swap]

    return matrix_new
