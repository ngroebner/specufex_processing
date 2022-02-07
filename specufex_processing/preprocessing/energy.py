import numpy as np
import scipy.signal as scs
from scipy.stats import entropy

def waveform_energy(waveform,method):
    """Calculate energy and entropy of a waveform
    Arguments:
        waveform (numpy.array): The waveform of interest
        method (str): One of 'hilbert_env' or 'abssquared'
    Returns:
        energy, entropy"""

    # amplitude envelope way (but this is not really energy units)
    if method=='hilbert_env':
        waveform_env = np.abs(scs.hilbert(waveform))
        waveform_env_n = waveform_env/np.max(waveform_env)
        enrgy = np.sum(waveform_env)/len(waveform_env_n)
    if method=='abssquared':
        waveform_env = (np.abs(waveform))**2
        waveform_env_n = waveform_env/np.max(waveform_env)
        enrgy = np.sum(waveform_env)
    return enrgy, entropy(waveform_env_n/np.sum(waveform_env_n))