# Specufex processing utilities

This document should eventually have an introduction and short tutorial on how to use the processing scripts. For now, it will serve as a place for developer notes and to document specifications such as for the wf_cat_out file.

## Testing

There is now a standard testing framework started.
Whenever you make a change to the code that you want to push, run the tests to see if any errors are generated.
For now, there is a single dataset used, but we can add more in the future for morre robust testing.
To run the tests, you must install the `pytest` library

```
>>> pip install pytest
```

After installation, to run the tests, simply cd to the "tests" directory and run

```
>>> pytest
```

If everything works, it will tell you that everything passed.
Currently, only the first 2 scripts (1_makeWaveformsDataset and 2_convertToSpectrograms) have tests.

## wf_cat.csv structure

The `wf_cat.csv` file is the catalog that these methods use for identifying the specific waveforms and their characteristics. The columns are listed below.

- _ev_ID_
  - The unique event ID for the waveform.
- _filename_
  - The name of the file containing the waveform
- _timestamp_
  - Timestamp in aribitrary units. Could be in datetime format, seconds, nanoseconds, etc, but the only requirement is that it designates time as a numer.
