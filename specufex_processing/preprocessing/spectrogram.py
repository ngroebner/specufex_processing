import h5py
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from specufex_processing.utils import (
    _overwrite_group_if_exists,
    _overwrite_dataset_if_exists
)

import tqdm

np.seterr(all='raise')

class SpectrogramMaker:

    def __init__(self,fs,nperseg,noverlap,nfft,
                       mode,scaling,
                       trim=False,fmin=None,fmax=None,norm_waveforms=True):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.scaling = scaling
        self.mode = mode
        self.trim = trim
        self.fmin = fmin
        self.fmax = fmax
        self.norm_waveforms=norm_waveforms

    def __call__(self, waveform, normalize=None):

        """Converts a waveform into a transformed spectrogram
        Returns
        -------
        tuple of 2 numpy arrays:
            STFT - the transformed spectrogram
            STFT_0 - the original, non-transformed spectrogram

        """

        if normalize == None :
            normalize = self.norm_waveforms
            #print(f'Using {self.norm_waveforms} for normalization')

        ##### Normalize each waveform #####
        if normalize:
            waveform = waveform / np.abs(waveform).max()

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
        if normConstant != 0:
            STFT_dB = 20*np.log10(STFT_0/normConstant, where=STFT_0 != 0)  ##convert to dB
        else:
            STFT_dB = 20*np.log10(STFT_0, where=STFT_0 !=0)
        STFT = np.maximum(0, STFT_dB)

        self.fSTFT = fSTFT
        self.tSTFT = tSTFT

        return STFT, STFT_0

    def save2hdf5(self, spects, raw_spects, evIDs, filename):
        """Save spectrograms and associated event IDs to standard H5 format."""

        with h5py.File(filename,'a') as fileLoad:

            spectrograms_group = _overwrite_group_if_exists(
                fileLoad,
                "spectrograms"
            )
            trans_spectrograms_group =_overwrite_group_if_exists(
                spectrograms_group,
                "transformed_spectrograms"
            )
            raw_spectrograms_group = _overwrite_group_if_exists(
                spectrograms_group,
                "raw_spectrograms"
            )

            # write spectrogram parameters as attributes
            spectrograms_group.attrs["fs"] = self.fs
            spectrograms_group.attrs["nperseg"] = self.nperseg
            spectrograms_group.attrs["noverlap"] = self.noverlap
            spectrograms_group.attrs["nfft"] = self.nfft
            spectrograms_group.attrs["mode"] = self.mode
            spectrograms_group.attrs["scaling"] = self.scaling
            spectrograms_group.attrs["fmin"] = self.fmin
            spectrograms_group.attrs["fmax"] = self.fmax

            # spectrogram axes
            _overwrite_dataset_if_exists(spectrograms_group, "fSTFT", self.fSTFT)
            _overwrite_dataset_if_exists(spectrograms_group, "tSTFT", self.tSTFT)

            for i, spect in enumerate(spects):
                trans_spectrograms_group.create_dataset(name=evIDs[i], data=spect)
            for i, rspect in enumerate(raw_spects):
                raw_spectrograms_group.create_dataset(name=evIDs[i], data=rspect)

            print(f"{len(trans_spectrograms_group)} spectrograms saved.")

def create_spectrograms(
    waveform_h5_path,
    station,
    channel,
    winLen_Sec,
    fracOverlap,
    nfft,
    fmin,
    fmax,
    norm_waveforms=True
):
    """Create spectrograms from h5 file
    """

    with h5py.File(waveform_h5_path,'r+') as fileLoad:
        # sampling rate, Hz - only oicks first one, a bad thing
        # TODO: guarantee upstream that all waveforms have same sampling rate
        fs = fileLoad[f"{station}/processing_info"].get('sampling_rate_Hz')[()]

        # spectrogram parameters
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        nperseg = int(winLen_Sec*fs) #datapoints per window segment
        noverlap = nperseg*fracOverlap  #fraction of window overlapped

        #padding must be longer than n per window segment
        if nfft < nperseg:
            nfft = nperseg*2
            print("nfft too short; changing to ", nfft)

        mode='magnitude'
        scaling='spectrum'

        spectmaker = SpectrogramMaker(
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            mode=mode,
            scaling=scaling,
            trim=True,
            fmin=fmin,
            fmax=fmax,
            norm_waveforms=norm_waveforms)

        badevIDs = []
        evIDs = []
        spects = []
        raw_spects = []
        for evID in tqdm.tqdm(fileLoad[f'waveforms/{station}/{channel}'].keys()):
            waveform = fileLoad[f'waveforms/{station}/{channel}/{evID}'][:]
            STFT, STFT_0 = spectmaker(waveform)

            if np.any(np.isnan(STFT)) \
                    or np.any(STFT)==np.inf \
                    or np.any(STFT)==-np.inf \
                    or STFT.sum()==0:
                badevIDs.append(evID)
                # if you get a bad one, fill with zeros
                # this is to preserve ordering
                STFT = np.zeros_like(STFT)
                print(f"evID {evID} set to zero, bad data")
            """if np.median(STFT_0) == 0:
                print(f"Zero median: evID {evID}")
            if np.all(STFT==0):
                print(f"STFT all zero: evID {evID}")"""
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
