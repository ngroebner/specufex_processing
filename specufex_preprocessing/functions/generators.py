#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 05:34:57 2021

@author: theresasawi
"""
import scipy as sp


import h5py
import numpy as np
import sys
import obspy
import os

sys.path.append('functions/')

import tables
tables.file._open_files.close_all()
import scipy.io as spio
import scipy.signal


#%%



def getEventID(path,key):
    """
    Generate unique event ID based on filename
    

    """

    
    if 'Parkfield' in key:

        evID = path.split('/')[-1].split('.')[-1]
        
    else:## default:: event ID is the waveform filename 
        
        evID  = path.split('/')[-1].split('.')[0]            
    

    return evID


#%%


def gen_wf_from_folder(wf_filelist,key,lenData,channel_ID):
    """
    Note
    ----------
   ** MAKE NEW FOR EACH DATASET:: Add settings for your project key below

    Parameters
    ----------
    wf_filelist : list of paths to waveforms
    key : project key name
    lenData : number of sampels in data (must be same for all data)
    channel_ID : for obspy streams, this is the index of the desired channel

    Yields
    ------
    data : np array of wf data
    evID : formatted event ID
    Nkept : number of kept wfs

    """

    Nkept=0 # count number of files kept
    Nerr = 0 # count file loading error
    NwrongLen = 0

    for i, path in enumerate(wf_filelist):

        
        evID = getEventID(path,key)
        
        try: #catch loading errors

            st = obspy.read(path)

            ### REMOVE RESPONSE ??

            st.detrend('demean')




            #####            #####            #####            #####
            ### MAKE NEW FOR EACH DATASET
            #####            #####            #####            #####
            if 'Parkfield' or 'GeysersNW' or 'GeysersNW_filt' in key:

                data = st[channel_ID].data

                
            #####            #####            #####            #####
                        #####            #####            #####            #####
            #####            #####            #####            #####


             #make sure data same length
            if len(data)==lenData:

                Nkept += 1

                yield data, evID, Nkept

                if i%100==0:
                    print(f"{i}/{len(wf_filelist)}")

            #Parkfield is sometimes one datapoint off
            elif np.abs(len(data) - lenData) ==1:

                data = data[:-1]
                Nkept += 1
                yield data, evID, Nkept

            else:
                NwrongLen += 1
                print(NwrongLen, ' data wrong length')
                print(f"this event: {len(data)}, not {lenData}")


        except ValueError: #some of the data are corrupt; unloadable
            Nerr +=1
            print(Nerr, ". File ", path, " unloadable")
            pass


#%% make sgram generator
def gen_sgram_QC(key,evID_list,dataH5_path,trim=True,saveMat=False,sgramOutfile='.',**args):

    fs=args['fs']
    nperseg=args['nperseg']
    noverlap=args['noverlap']
    nfft=args['nfft']
    mode=args['mode']
    scaling=args['scaling']
    fmin=args['fmin']
    fmax=args['fmax']
    Nkept = 0
    evID_BADones = []
    for i, evID in enumerate(evID_list):

        if i%100==0:
            print(i,'/',len(evID_list))

        with h5py.File(dataH5_path,'a') as fileLoad:
            stations=args['station']
            data = fileLoad[f"waveforms/{stations}/{args['channel']}"].get(str(evID))[:]



        fSTFT, tSTFT, STFT_raw = sp.signal.spectrogram(x=data,
                                                    fs=fs,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    #nfft=Length of the FFT used, if a zero padded FFT is desired
                                                    nfft=nfft,
                                                    scaling=scaling,
                                                    axis=-1,
                                                    mode=mode)

        if trim:
            freq_slice = np.where((fSTFT >= fmin) & (fSTFT <= fmax))
            #  keep only frequencies within range
            fSTFT   = fSTFT[freq_slice]
            STFT_0 = STFT_raw[freq_slice,:][0]
        else:
            STFT_0 = STFT_raw
            # print(type(STFT_0))


        # =====  [BH added this, 10-31-2020]:
        # Quality control:
        if np.isnan(STFT_0).any()==1 or  np.median(STFT_0)==0 :
            if np.isnan(STFT_0).any()==1:
                print('OHHHH we got a NAN here!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass
            if np.median(STFT_0)==0:
                print('OHHHH we got a ZERO median here!!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass

        if np.isnan(STFT_0).any()==0 and  np.median(STFT_0)>0 :

            normConstant = np.median(STFT_0)

            STFT_norm = STFT_0 / normConstant  ##norm by median

            STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0)  ##convert to dB
            # STFT_shift = STFT_dB + np.abs(STFT_dB.min())  ##shift to be above 0
    #

            STFT = np.maximum(0, STFT_dB) #make sure nonnegative


            if  np.isnan(STFT).any()==1:
                print('OHHHH we got a NAN in the dB part!')
                evID_BADones.append(evID)
                pass
            # =================save .mat file==========================

            else:

                Nkept +=1

                if saveMat==True:
                    if not os.path.isdir(sgramOutfile):
                        os.mkdir(sgramOutfile)


                    spio.savemat(sgramOutfile + str(evID) + '.mat',
                              {'STFT':STFT,
                                'fs':fs,
                                'nfft':nfft,
                                'nperseg':nperseg,
                                'noverlap':noverlap,
                                'fSTFT':fSTFT,
                                'tSTFT':tSTFT})


            # print(type(STFT))

            yield evID,STFT,fSTFT,tSTFT, normConstant, Nkept,evID_BADones, i
