import os


import pandas as pd
import numpy as np
import h5py
import torch

from tqdm import tqdm

SEIZURE_STRINGS = ['sz','seizure','absence','spasm']
FREQUENCY = 200
EPOCH_LENGTH_SEC = 10
INCLUDED_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz',
'EEG Pg1', 'EEG Pg2', 'EEG A1', 'EEG A2', 'EEG FT9', 'EEG FT10']
# PATH_TO_DATA = "../../../jdunnmon/EEG/eegdbs/SEC/stanford/"
PATH_TO_DATA = "../../../jdunnmon/data/EEG/eegdbs/SEC/stanford/"
PATH_TO_FILENAMES = "file_markers/"


from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from random import shuffle # used to shuffle initial file names
from random import randint # used to randomly pull nonSeizures

""" Hershey edits
Now, we're just loading up the data set by calling get_seizures.getDataLoaders()
and then returning indices and the length according to the items returned from that
I left fetch_dataloader as is, so it just creates the same train, test, and dev sets"""

def getOrderedChannels(labelsObject):
    labels = list(labelsObject)
    for i in range(0, len(labels)): # needed because might come as b strings....
        labels[i] = labels[i].decode("utf")
    orderedChannels = []
    for ch in INCLUDED_CHANNELS:
        orderedChannels.append(labels.index(ch))
    return orderedChannels

def sliceEpoch(orderedChannels, signals, sliceTime):
    if (sliceTime == -1):
        maxStart = max(signals.shape[1] - FREQUENCY * EPOCH_LENGTH_SEC, 0)
        sliceTime = randint(0, maxStart)
        sliceTime /= FREQUENCY
    startTime = int(FREQUENCY * sliceTime)
    endTime = int(FREQUENCY * (sliceTime + EPOCH_LENGTH_SEC))
    sliceMatrix = signals[orderedChannels,  startTime : endTime]

    # standardize by row
    row_sums = sliceMatrix.sum(axis=1)
    row_sums = row_sums[:, np.newaxis]
    sliceMatrix = sliceMatrix * 1.0 / row_sums

    # pad if necessary
    diff = FREQUENCY * EPOCH_LENGTH_SEC - sliceMatrix.shape[1]
    if diff > 0:
        zeros = np.zeros((sliceMatrix.shape[0], diff))
        sliceMatrix = np.concatenate((sliceMatrix, zeros))

    sliceMatrix = sliceMatrix.T
    return sliceMatrix

def getSeizureTimes(hdf):
    annot = hdf['record-0']['edf_annotations']
    antext = [s.decode('utf-8') for s in annot['texts'][:]]
    starts100ns = [xx for xx in annot['starts_100ns'][:]]
    df = pd.DataFrame(data=antext, columns=['text'])
    df['starts100ns'] = starts100ns
    df['starts_sec'] = df['starts100ns']/10**7
    seizureDF = df[df.text.str.contains('|'.join(SEIZURE_STRINGS),case=False)]
    seizureTimes = seizureDF['starts_sec'].tolist()
    return seizureTimes

def parseTxtFiles(seizure_file, nonseizure_file):
    seizure_contents = open(seizure_file, "r")
    seizure_str = seizure_contents.readlines()

    nonseizure_contents = open(nonseizure_file, "r")
    nonseizure_str = nonseizure_contents.readlines()

    smaller_half = min(len(seizure_str), len(nonseizure_str))

    combined_str = seizure_str[ : smaller_half] + nonseizure_str[ : smaller_half]

    shuffle(combined_str)

    combined_tuples = []
    for i in range(len(combined_str)):
        tup = combined_str[i].strip("\n").split(",")
        tup[1] = int(tup[1])
        combined_tuples.append(tup)

    return combined_tuples

class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    

    def __init__(self, data_dir, split_type):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.file_tuples = parseTxtFiles(PATH_TO_FILENAMES + "seizures_marked.txt", 
                                            PATH_TO_FILENAMES + "nonSeizures_marked.txt")
        total = 10 #len(self.file_tuples)
        if split_type == 'train':
            self.file_tuples = self.file_tuples[ : int(total * 0.8)]
        elif split_type == 'val':
            self.file_tuples = self.file_tuples[int(total * 0.8) : int(total * 0.9)]
        elif split_type == 'test':
            self.file_tuples = self.file_tuples[int(total * 0.9) : int(total)]

        self.data_dir = data_dir
        self.lastSeizure = (torch.zeros(2000, 25), 0)
        self.num_seiz_shaped = 0
        self.num_nonseiz_shaped = 0

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        file_name, seizure_idx = self.file_tuples[idx]

        currentFileName = self.data_dir + file_name
        hdf = h5py.File(currentFileName)
        orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])

        if orderedChannels == None:
            return self.lastSeizure

        if seizure_idx == -1:
            nonSeizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], -1)
            nonSeizure = torch.FloatTensor(nonSeizure)
            return (nonSeizure, 0)
        else:
            seizureTimes = getSeizureTimes(hdf)
            seizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], seizureTimes[seizure_idx])
            seizure = torch.FloatTensor(seizure)
            self.lastSeizure = (seizure, 1)
            return self.lastSeizure


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(PATH_TO_DATA, split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(PATH_TO_DATA, split), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

if __name__ == '__main__':
    PATH_TO_DATA = "../../../../jdunnmon/data/EEG/eegdbs/SEC/stanford/"
    PATH_TO_FILENAMES = "../file_markers/"
    sd = SIGNSDataset(PATH_TO_DATA, 'train')
    for i, (train_batch, labels_batch) in enumerate(sd):
        print(i, train_batch.shape)
        # print(sd.num_seiz_shaped, sd.num_nonseiz_shaped)

    # sd = SIGNSDataset(PATH_TO_DATA, 'train')
    # print(len(sd))
    # sd = SIGNSDataset(PATH_TO_DATA, 'val')
    # print(len(sd))
    # sd = SIGNSDataset(PATH_TO_DATA, 'test')
    # print(len(sd))
