import numpy as np
import pandas as pd
import obspy
from sklearn.preprocessing import MinMaxScaler,RobustScaler,MaxAbsScaler,StandardScaler
import matplotlib.pyplot as plt

def dataframe2hdf(df, group):
    """Saves a pandas DataFrame to a group in an hdf5 file.

    Arguments:
    ----------
    df: pandas.DataFrame
        DataFrame to save
    group: h5py group
        Group object.

    Returns
    -------
    Nothing

    """

    for col in df:
        if df[col].dtype == 'object':
            group.create_dataset(
                name=col,
                data=np.array(df[col],dtype='S'))
        else:
            group.create_dataset(name=col, data=df[col])

def normalize_waveform(waveforms,norm_scale='MaxAbsScaler',save_plot=True,path_to_save=None):
    if norm_scale == 'MaxAbsScaler' :
        scaler = MaxAbsScaler()
    elif norm_scale == 'RobustScaler' :
        scaler = RobustScaler()
    elif norm_scale == 'StandardScaler' :
        scaler = StandardScaler()
    elif norm_scale == 'MinMaxScaler':
        scaler = MinMaxScaler((-1,1))
    else :
        print(f'Waveform normalization {norm_scale} not recognized, please use MaxAbsScaler, MinMaxScaler((-1,1)),RobustScaler,StandardScaler')
    scaler.fit(np.transpose(waveforms- np.mean(waveforms, axis=1)[:,np.newaxis]))
    norm_waveforms = np.transpose(scaler.transform(np.transpose(waveforms)))

    if save_plot :
        plt.ioff()
        plt.figure(figsize=(15, 14))
        plt.title(f"Input dataset")
        plt.pcolormesh(norm_waveforms,cmap=plt.cm.RdBu)
        plt.colorbar(label='Normalized Amplitude')
        plt.xlabel("Waveform Time")
        plt.ylabel("Waveform Count")
        plt.tight_layout()
        plt.savefig(path_to_save+'_Plot_Waveform.png')
        plt.close()
        plt.ion()

    return norm_waveforms


def load_wf(filename, lenData, channel_ID=None):
    """Loads a waveform file and returns the data.

    Arguments
    ---------
    filename: str
        Filename to load
    lenData: int
        Number of samples in the file. Must be uniform for all files.
    channel_ID: int
        If the fileis an obspy stream, this is the desired channel.
    """
    if ".txt" in filename:
        data = np.loadtxt(filename)
    else: #catch loading errors
        st = obspy.read(filename)
        ### REMOVE RESPONSE ??
        st.detrend('demean')
        data = st[channel_ID].data

    #make sure data same length
    Nkept = 0
    if len(data)==lenData:
        return data
    #Parkfield is sometimes one datapoint off
    elif np.abs(len(data) - lenData) ==1:
        data = data[:-1]
        Nkept += 1
        return data

    else:
        print(filename, ': data wrong length')
        print(f"this event: {len(data)}, not {lenData}")
        return None

def getEventID(path,key,eventID_string):
    """
    Generate unique event ID based on filename


    """
    evID = eval(eventID_string)
    # print(evID)

    # if 'Parkfield' in key:
    #
    #     evID = path.split('/')[-1].split('.')[-1]
    #
    # else:## default:: event ID is the waveform filename
    #
    #     evID  = path.split('/')[-1].split('.')[0]


    return evID
