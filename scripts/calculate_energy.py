#!python3
# -*- coding: utf-8 -*-

import argparse
import os

import h5py
import yaml

from specufex_processing.preprocessing.energy import waveform_energy
from specufex_processing.utils import _overwrite_group_if_exists

parser = argparse.ArgumentParser()
parser.add_argument("config_filename", help="Path to configuration file.")
args = parser.parse_args()

# load config file
with open(args.config_filename, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# pull out config values for conciseness
path_config = config["paths"]
key = path_config["key"]
data_config = config['dataParams']
station = data_config["station"]
channel = data_config["channel"]

# build path strings
dataH5_name = f'data_{key}.h5'
projectPath = path_config["projectPath"]
dataH5_path = os.path.join(projectPath,'H5files/', dataH5_name)

with h5py.File(dataH5_path,'a') as h5file:

    energy_group = _overwrite_group_if_exists(h5file, "energy")
    energy_channel_grp = h5file.create_group(f"energy/{station}/{channel}")
    entropy_group = _overwrite_group_if_exists(h5file, "entropy")
    entropy_channel_grp = h5file.create_group(f"entropy/{station}/{channel}")

    evIDs = list(h5file["waveforms"][station][channel].keys())

    print("Calculating energy and entropy of each waveform")
    for evID in evIDs:
        waveform = h5file["waveforms"][station][channel][evID]
        energy, entropy = waveform_energy(waveform, "abssquared")
        energy_channel_grp.create_dataset(name=evID, data=energy)
        entropy_channel_grp.create_dataset(name=evID, data=entropy)

print(f"Done. Saved to {dataH5_path}")
