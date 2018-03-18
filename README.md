# Seizure Detection from EEG Data

*Author: Nick Hershey*

## Setup

To install all required items
```
pip install -r requirements.txt
```

## Task

Given an EEG waveform of 10 seconds, identify it as being part of a seizure or not.


## Files

1. __get_seizures.py__: Reads .eeghdf files from the specified directory (`--data_dir`) and writes tuples of the seizure file and an index to `marked_files/seizures.txt` and `marked_files/nonSeizures.txt`. The index is -1 if the file contains no seizures or 0 to `n` to if there are `n` seizures in that file. That is, for a file with `n` seizures, there will be `n` entries of "file,i" in the txt file. These files are then read in `model/data_loader.py`.
```
python get_seizures.py --data_dir {where data is stored}
```

2. __model/data_loader.py__:  Loads in an EEG slice asynchronously from a text file in file_markers. Currently, we are pulling epochs of 10 seconds at a frequency of 200 Hz on the 25 standard EEG nodes. This creates a 25x2000 input.

3. __experiments__: Contains the parameters and results of the various models and hyperparameter tests.

4. __model/net.py__: Contains the neural network architectures we are using. Currently there are three:
    a. base: flatten the matrix and apply a neural net with one hidden layer
    b. conv: apply a standard vision neural net of 3 convolutional layers (filters, batch norm, pool, relu)
and then flatten and apply two fully connected layers with batch norm and dropout
    c. lstm: 2-layers of LSTM units for each output, run the last hidden output through a fully connected layer
    It also contains the loss function, accuracy, and f1 score definitions.

4. __train.py__: Trains the neural network
```
python train.py --data_dir {where data is stored} --model_dir experiments/{base,conv,lstm}_model`
```

5. __evaluate.py__: Used after each epoch of training to test each of the accuracies

6. __search_hyperparams.py__: Used to test a parameter defined in one of the experiments. I haven't really run any yet.
```
python search_hyperparams.py --data_dir {where data is stored} --parent_dir experiments/{learning_rate,filter_size,etc.}
```

7. __synthesize_results.py__: Display the results of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

## Results
Stay tuned...
