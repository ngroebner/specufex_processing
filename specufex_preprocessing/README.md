# Specufex processing utilities

## Quickstart

There are three steps in the preprocessing workflow for Specufex:

1. Given a catalog of of events, a directory of waveform files corresponding to those events, and a configuration file, the waveforms are saved in a standarrd formnat to an hdf5 file ([1_makeWavefromsDataset.py](1_makeWaveformsDataset.py)).

2. The waveforms are then converted into spectrograms, and the spectrograms are saved to hdf5 ([2_convertToSpectrograms.py](2_convertToSpectrograms.py)).

3. Specufex is run on the resulting spectrograms, and fingerprints, etc, are written to a standardized hdf5file.

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
