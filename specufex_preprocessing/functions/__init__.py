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
    import numpy as np

    for col in df:
        if df[col].dtype == 'object':
            group.create_dataset(
                name=col,
                data=np.array(df[col],dtype='S'))
        else:
            group.create_dataset(name=col, data=df[col])