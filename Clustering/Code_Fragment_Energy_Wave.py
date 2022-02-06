import scipy.signal as scs
from scipy.stats import entropy

def waveform_energy(waveform,method):
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

## Number of points to resample the waveforms to 
resampl_size = 1000 

# Combine all waveforms into one file in order to run specufex on all
# first extract the data from each individual spectrogram h5

evIDs = []
waveforms = []
normconst = []
experiment = []
catalog = pd.DataFrame()

evIDs_cat = []
evIDs = []
evIDs = []
enrgy_all = []
entropy_all = []

files = [
      "../../00_Data/Dry_Basalt/data_Dry_AEs.h5",
      "../../00_Data/Wet_Basalt/data_wet_AEs.h5",
      "../../00_Data/Dry_Basalt_Ar/data_bs_dry.h5",
     "../../00_Data/Wet_Basalt_CO2/data_co2_AEs.h5",
    "../../00_Data/Wet_Basalt_CO2_B/data_or3b_aes.h5",
]

# f = h5py.File(files[0], "r")
# f['waveforms/pzt01'].keys()

for file in files:
    print(file)
    with h5py.File(file, "r") as f:
        try:
            evIDs_ = list(f["waveforms/1/na"].keys())
            for evID in evIDs_:
                waveform = resample(f["waveforms/1/na"][evID][()], resampl_size)
                
                enrgy, entrpy = waveform_energy(waveform,'abssquared')
                waveforms.append(waveform)
                exp_name = file.split("_")[-2]
                if exp_name == "OR2": exp_name="CO2+water"
                if exp_name == "bs": exp_name="dry_argon"
                experiment.append(exp_name)
                enrgy_all.append(enrgy)
                entropy_all.append(entrpy)
            evIDs += evIDs_
        except:
            try:
                evIDs_ = list(f["waveforms/pzt01/ch01"].keys())
                for evID in evIDs_:
                    waveform = resample(f["waveforms/pzt01/ch01"][evID][()],resampl_size )
                    enrgy, entrpy = waveform_energy(waveform,'abssquared')
                    waveforms.append(waveform)
                    exp_name = file.split("_")[-2]
                    if exp_name == "OR2": exp_name="CO2+water"
                    if exp_name == "bs": exp_name="dry_argon"
                    experiment.append(exp_name)
                    enrgy_all.append(enrgy)
                    entropy_all.append(entrpy)
                evIDs += evIDs_
            except:
                try :
                    evIDs_ = list(f["waveforms/pzt01/01"].keys())
                    for evID in evIDs_:
                        waveform = resample(f["waveforms/pzt01/01"][evID][()], resampl_size)
                        enrgy, entrpy = waveform_energy(waveform,'abssquared')
                        waveforms.append(waveform)
                        exp_name = file.split("_")[-2]
                        if exp_name == "OR2": exp_name="CO2+water"
                        if exp_name == "bs": exp_name="dry_argon"
                        experiment.append(exp_name)
                        enrgy_all.append(enrgy)
                        entropy_all.append(entrpy)
                    evIDs += evIDs_
                except:
                    evIDs_ = list(f["waveforms/pzt01/1"].keys())
                    for evID in evIDs_:
                        waveform = resample(f["waveforms/pzt01/1"][evID][()],resampl_size )
                        enrgy, entrpy = waveform_energy(waveform,'abssquared')
                        waveforms.append(waveform)
                        exp_name = file.split("_")[-2]
                        if exp_name == "or3b": exp_name="CO2_water_Open"
                        experiment.append(exp_name)
                        enrgy_all.append(enrgy)
                        entropy_all.append(entrpy)
                    evIDs += evIDs_
    
    #         df_ = pd.DataFrame()
    #         try:
    #             print(f["catalog/cat_by_sta/pzt01"].keys())
    #             for col in f["catalog/cat_by_sta/pzt01"].keys():
    #                 df_[col] = f["catalog/cat_by_sta/pzt01"][col][()].astype(str)
    #         except:
    #             try :
    #                 print(f["catalog/cat_by_sta/"].keys())
    #                 for col in f["catalog/cat_by_sta/1"].keys():
    #                     df_[col] = f["catalog/cat_by_sta/1"][col][()]
    #                     df_.rename(columns={"ev_ID": "event_ID"},inplace=True)
    #             except :
    #                 print(f["catalog/"].keys())
    #                 for col in f["catalog/"].keys():
    #                     df_[col] = f["catalog/"][col][()]
    #         catalog = pd.concat([catalog, df_])

