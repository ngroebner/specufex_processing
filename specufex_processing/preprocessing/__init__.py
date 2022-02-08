import numpy as np
import pandas as pd
import obspy

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
    print(filename)
    if ".txt" in filename:
        data = np.loadtxt(filename)
    elif ".npy" in filename:
        data = np.load(filename)
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