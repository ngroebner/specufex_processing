# Specufex processing utilities

## Quickstart

There are three steps in the preprocessing workflow for Specufex:

1. Given a catalog of of events, a directory of waveform files corresponding to those events, and a configuration file, the waveforms are saved in a standard format to an hdf5 file ([1_makeWavefromsDataset.py](1_makeWaveformsDataset.py)).

1a. *Optional:* Calculate energy and entropy for your individual waveforms ([calculate_energy.py](calculate_energy.py))

2. The waveforms are then converted into spectrograms, and the spectrograms are saved to hdf5 ([2_convertToSpectrograms.py](2_convertToSpectrograms.py)).

3. Specufex is run on the resulting spectrograms, and fingerprints, etc, are written to a standardized hdf5file.

To get started, create a catalog for the events of interest in the standard catalog structure below. The catalog is very simple, and contains the minimum information needed to run specufex. As this is a minimum specification, your catalog can contain additional data columns, but must contain at least _ev_ID_, _filename_, and _timestamp_.

Next, create a config file with the parameters needed for the run. An example config file is [here](example_config.yml).

Convert the folder of waveform files into the standard specufex data format, replacing ```example_config.yaml``` with the name of your config file:

``` bash
>>> 1_makeWaveformsDataset.py example_config.yaml
```

This will save an hdf5 file containing the waveforms to the path specified in your config file.

> *Optional*
>
> If you need energy and/or entropy calculations, run the following to calculate these and save the values to the ```data_projname.h5``` file.
>
> ``` bash
> >>> calculate_energy.py example_config.py
> ```

Next, run the following to convert the waveforms to spectrograms.

``` bash
>>> 2_convertToSpectrograms.py example_config.py
```

Finally, the following will run specufex on the spectrograms and calculate fingerprints. Note that this will take considerably longer than the last two steps.

``` bash
>>> 3_runSpecUFEx.py example_config.yaml
```

The output will be saved to the directory specified in your config file.

```
|--results_directory
   |-- H5files
       |-- data_projname.h5
       |-- SpecUFEx_projname.h5
   |-- sgram_cat_out_projname.csv
   |-- wf_cat_out.csv
```

These are specific file structures for  [SpecUFEx_projname.h5](specufex_projname-structure.md) and [data_projname.h5](data_projname-structure.md).

Some notes: May want to separate the nmf part from the hmm part, so that hmm can be run with several different parameter combos without rerunning nmf.

## Standard catalog structure

The `wf_cat.csv` file is the catalog that these methods use for identifying the specific waveforms and their characteristics. The columns are listed below.

- _ev_ID_
  - The unique event ID for the waveform.
- _filename_
  - The name of the file containing the waveform
- _timestamp_
  - Timestamp in aribitrary units. Could be in datetime format, seconds, nanoseconds, etc, but the only requirement is that it designates time as a number.

## Development

See [development documentation](Development.md) for more details.
