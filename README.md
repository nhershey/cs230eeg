# Seizure Detection from EEG Data

*Author: Nick Hershey*

## Setup

To install all required items
```
pip install -r requirements.txt
```

## Task

Given an EEG waveform of 10 seconds, identify it as being part of a seizure or not.


## Changes

1. __get_seizures.py__: Reads .eeghdf files from the specified directory and outputs a list of tensors representing seizure and non-seizure EEG  t-second waveforms with their appropriate labels. This works by searching the hdf file for annotations related to the start of a seizure and then slicing a t-second section from the EEG reading. Currently, we are pulling epochs of 10 seconds at a frequency of 200 Hz on the 25 standard EEG nodes. This creates a 25x2000 input. For this dummy baseline, I flatten this vector into a 50,000-dimensional vector.

2. __model/data_loader.py__: Modified it to read the data pulled from `get_seizures.py`. We are currenly running on 10 EEG files with one seizure found, so we simply make the dev, train, and test sets identical.

3. __build_dataset.py__: Is now unused, as all processing occurs in `get_seizures.py`.

4. __model/net.py__: I modified the net.py to work with this differently sized data. I currently flatten the EEG 10-second 25-channel waveform into a 50,000-dimensional vector and run it through a fully-connected, feed-forward neural network with one hidden layer with 100 units. The hidden layer uses a ReLU activation function whereas the final layer uses a sigmoid with logistic loss.

5. __evaluate.py, search_hyperparams.py, synthesize_results.py__: Each of these files is unmodified.

## Results
We are currently running on just the 10 eeghdf files in `data/dummy_data`, which contains only 1 seizure, so we trivially reach 100% accuracy.
