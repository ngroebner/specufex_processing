# Specufex processing utilities

This document should eventually have an introduction and short tutorial on how to use the processing scripts. For now, it will serve as a place to document to specifications such as for the wf_cat_out file.

## wf_cat.csv structure

The `wf_cat.csv` file is the catalog that these methods use for identifying the specific waveforms and their characteristics. The columns are listed below.

- _ev_ID_
  - The unique event ID for the waveform.
- _filename_
  - The name of the file containing the waveform
- _timestamp_
  - Timestamp in aribitrary units. Could be in datetime format, seconds, nanoseconds, etc, but the only requirement is that it designates time as a numer.
