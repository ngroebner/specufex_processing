# SpecUFEx_projname.h5 data structure

The preprocessing workflow generates an HDF5 file that contains the intermediate and final output of the steps. Below is the structure of that file.

- ***SpecUFEx_projname.h5***

  - **ACM** (group)  
  Activation coefficient matrices
    - **evID_0** (group): *dataset*
    - .......
    - **evID_n** (group): *dataset*
<br> <br>

  - **STM** (group)  
  State time matrix
    - **evID_0** (group): *dataset*
    - .......
    - **evID_n** (group): *dataset*
<br> <br>

  - **SpecUFEx_output** (group)
    - *ACM_gain* (dataset):  
    Activation coefficient matrix gain vector
    - **As** (group):  
    A matrices from HMM step
      - **evID_0** (group): *dataset*
      - .......
      - **evID_n** (group): *dataset*
<br> <br>

  - **catalog** (group):  
  Copy of the event catalog. Can load into Pandas DataFrame
<br> <br>

  - **fingerprints** (group):  
  the fingerprint vectors, saved in square format
    - **evID_0** (group): *dataset*
    - .......
    - **evID_n** (group): *dataset*
<br> <br>

  - **model_parameters** (group):  
  The fitted matrices for the NMF and HMM models
    - *EA* (dataset)
    - *EB* (dataset)
    - *EW* (dataset)
    - *ElnWA* (dataset)
<br> <br>

  - **spectrograms** (group)
    - **attrs** (attributes)
      - *fs*: sampling frequency
      - *nfft*: number of fft in the spectrogram
      - *noverlap*: number of overlaps in segment
      - *nperseg*: number of sample per segment
      - *mode*: spectrogram mode
      - *scaling*: spectrogram scaling
      - *fmax*: max frequency if spectrogram is trimmed
      - *fmin*: min frequency if spectrogram is trimmed
    - **raw_spectrograms** (group):  
    Spectrograms before the dB transformation
      - **evID_0** (group): *dataset*
      - .......
      - **evID_n** (group): *dataset*
<br> <br>

  - **transformed_spectrograms** (group):  
      Spectrograms after the dB transformation (input for the NMF step)
    - **evID_0** (group): *dataset*
    - .......
    - **evID_n** (group): *dataset*

