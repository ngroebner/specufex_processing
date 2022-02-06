import h5py
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from specufex_processing.utils import _overwrite_group_if_exists

# f

class SpectrogramMaker:

    def __init__(self,fs,nperseg,noverlap,nfft,
                       mode,scaling,
                       trim=False,fmin=None,fmax=None):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.scaling = scaling
        self.mode = mode
        self.trim = trim
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, waveform):
        """Converts a waveform into a transformed spectrogram
        Returns
        -------
        tuple of 2 numpy arrays:
            STFT - the transformed spectrogram
            STFT_0 - the original, non-transformed spectrogram

        """
        fSTFT, tSTFT, STFT_raw = spectrogram(waveform,fs=self.fs,nperseg=self.nperseg,
                                            noverlap=self.noverlap,nfft=self.nfft,
                                            scaling=self.scaling,axis=-1,mode=self.mode)
        if self.trim:
            freq_slice = np.where((fSTFT >= self.fmin) & (fSTFT <= self.fmax))
            #  keep only frequencies within range
            fSTFT   = fSTFT[freq_slice]
            STFT_0 = STFT_raw[freq_slice,:][0]
        else:
            STFT_0 = STFT_raw
        # convert
        normConstant = np.median(STFT_0)
        STFT_dB = 20*np.log10(STFT_0/normConstant, where=STFT_0 != 0)  ##convert to dB
        STFT = np.maximum(0, STFT_dB)

        self.fSTFT = fSTFT
        self.tSTFT = tSTFT

        return STFT, STFT_0

    def save2hdf5(self, spects, raw_spects, evIDs, filename):
        """Save spectrograms and associated event IDs to standard H5 format."""

        with h5py.File(filename,'a') as fileLoad:
            """if 'spectrograms' in fileLoad.keys():
                del fileLoad["spectrograms"]
            spectrograms_group = fileLoad.create_group(f"spectrograms")"""
            spectrograms_group = _overwrite_group_if_exists(fileLoad, "spectrograms")
            raw_spectrograms_group = _overwrite_group_if_exists(fileLoad, "raw_spectrograms")

            for i, spect in enumerate(spects):
                #print(evIDs[i])
                spectrograms_group.create_dataset(name=evIDs[i], data=spect)
            for i, rspect in enumerate(raw_spects):
                raw_spectrograms_group.create_dataset(name=evIDs[i], data=rspect)

            if 'spec_parameters' in fileLoad.keys():
                del fileLoad["spec_parameters"]

            spec_parameters_group  = fileLoad.create_group(f"spec_parameters")
            spec_parameters_group.clear()
            spec_parameters_group.create_dataset(name= 'fs', data=self.fs)
            #spec_parameters_group.create_dataset(name= 'lenData', data=self.lenData)
            spec_parameters_group.create_dataset(name= 'nperseg', data=self.nperseg)
            spec_parameters_group.create_dataset(name= 'noverlap', data=self.noverlap)
            spec_parameters_group.create_dataset(name= 'nfft', data=self.nfft)
            spec_parameters_group.create_dataset(name= 'mode', data=self.mode)
            spec_parameters_group.create_dataset(name= 'scaling', data=self.scaling)
            spec_parameters_group.create_dataset(name= 'fmin', data=self.fmin)
            spec_parameters_group.create_dataset(name= 'fmax', data=self.fmax)
            spec_parameters_group.create_dataset(name= 'fSTFT', data=self.fSTFT)
            spec_parameters_group.create_dataset(name= 'tSTFT', data=self.tSTFT)

            print(f"{len(spectrograms_group)} spectrograms saved.")

def create_spectrograms(
    waveform_h5_path,
    station,
    channel,
    winLen_Sec,
    fracOverlap,
    nfft,
    fmin,
    fmax,
    ):
    """Create spectrograms from h5 file
    """

    with h5py.File(waveform_h5_path,'r+') as fileLoad:
        # sampling rate, Hz - only oicks first one, a bad thing
        # TODO: guarantee upstream that all waveforms have same sampling rate
        print(fileLoad.keys())
        fs = fileLoad[f"{station}/processing_info"].get('sampling_rate_Hz')[()]
        # number of datapoints
        #lenData = fileLoad[f"{station}/processing_info"].get('lenData')[()]

        # spectrogram parameters
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        nperseg = int(winLen_Sec*fs) #datapoints per window segment
        noverlap = nperseg*fracOverlap  #fraction of window overlapped
        print(noverlap, nperseg)

        #padding must be longer than n per window segment
        if nfft < nperseg:
            nfft = nperseg*2
            print("nfft too short; changing to ", nfft)

        mode='magnitude'
        scaling='spectrum'
        # set args for generator

        spectmaker = SpectrogramMaker(fs=fs,nperseg=nperseg,
                            noverlap=noverlap,nfft=nfft,
                            mode=mode,scaling=scaling,
                            trim=True,fmin=fmin,fmax=fmax)

        badevIDs = []
        evIDs = []
        spects = []
        raw_spects = []
        for evID in fileLoad[f'waveforms/{station}/{channel}'].keys():
            #print(fileLoad[f'waveforms/{station}/{channel}'].keys())
            #print(fileLoad[f'waveforms/{station}/{channel}'].keys())
            waveform = fileLoad[f'waveforms/{station}/{channel}/{evID}'][:]
            STFT, STFT_0 = spectmaker(waveform)
            if np.any(np.isnan(STFT)) or np.any(STFT)==np.inf or np.any(STFT)==-np.inf:
                badevIDs.append(evID)
                # if you get a bad one, fill with zeros
                # this is to preserve ordering otherwise things
                # get f'd up
                STFT = np.zeros_like(STFT)
                print(f"evID {evID} set to zero, bad data")
            evIDs.append(evID)
            spects.append(STFT)
            raw_spects.append(STFT_0)
        print('N events in badevIDs: ', len(badevIDs))

        return evIDs, spects, raw_spects, spectmaker

def pad_spects(spects):
    # pad short spectrograms with zeros
    max_len = np.max(np.array([x.shape[1] for x in spects]))
    padded_spects = []
    for spect in spects:
        if spect.shape[1] < max_len:
            print("padding")
            npad = max_len - spect.shape[1]
            spect = np.pad(spect, pad_width=((0,0),(0,npad)))
            padded_spects.append(spect)
        else:
            padded_spects.append(spect)
    return padded_spects
