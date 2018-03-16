import random
import os


import pandas as pd
import numpy as np
import h5py
import torch

SEIZURE_STRINGS = ['sz','seizure','absence','spasm']
FREQUENCY = 200
EPOCH_LENGTH_SEC = 10
INCLUDED_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz',
'EEG Pg1', 'EEG Pg2', 'EEG A1', 'EEG A2', 'EEG FT9', 'EEG FT10']
# PATH_TO_DATA = "../../../jdunnmon/EEG/eegdbs/SEC/stanford/"
PATH_TO_DATA = "../../../jdunnmon/data/EEG/eegdbs/SEC/stanford/"
# INCLUDED_CHANNELS = ['EEG C3']


from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

""" Hershey edits
Now, we're just loading up the data set by calling get_seizures.getDataLoaders()
and then returning indices and the length according to the items returned from that
I left fetch_dataloader as is, so it just creates the same train, test, and dev sets"""

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

def getOrderedChannels(labelsObject):
    labels = list(labelsObject)
    for i in range(0, len(labels)): # needed because might come as b strings....
        labels[i] = labels[i].decode("utf")
    orderedChannels = []
    for ch in INCLUDED_CHANNELS:
        try:
            orderedChannels.append(labels.index(ch))
        except:
            # print("failed to get channel " + ch, labels)
            return None
    return orderedChannels

def sliceEpoch(orderedChannels, signals, sliceTime):
    startTime = int(FREQUENCY * sliceTime)
    endTime = int(FREQUENCY * (sliceTime + EPOCH_LENGTH_SEC))
    sliceMatrix = signals[orderedChannels,  startTime : endTime]
    sliceMatrix = sliceMatrix.T
    # standardize by row
    # row_sums = sliceMatrix.sum(axis=1)
    # sliceMatrix = sliceMatrix / row_sums[:, np.newaxis]
    return sliceMatrix


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
        self.filenames = os.listdir(data_dir)
        total = 10 #len(self.filenames)
        if split_type == 'train':
            self.filenames = self.filenames[ : int(total * 0.8)]
        elif split_type == 'val':
            self.filenames = self.filenames[int(total * 0.8) : int(total * 0.9)]
        elif split_type == 'test':
            self.filenames = self.filenames[int(total * 0.9) : int(total)]

        self.data_dir = data_dir
        self.lastSeizure = (torch.zeros(2000, 25), 0)

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        currentFileName = self.data_dir + self.filenames[idx]
        hdf = h5py.File(currentFileName)
        annot = hdf['record-0']['edf_annotations']
        antext = [s.decode('utf-8') for s in annot['texts'][:]]
        starts100ns = [xx for xx in annot['starts_100ns'][:]]
        df = pd.DataFrame(data=antext, columns=['text'])
        df['starts100ns'] = starts100ns
        df['starts_sec'] = df['starts100ns']/10**7
        seizureDF = df[df.text.str.contains('|'.join(SEIZURE_STRINGS),case=False)]
        orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])
        if orderedChannels == None:
            return self.lastSeizure

        if seizureDF.empty:
            nonSeizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], random.randint(0,1.0))
            nonSeizure = torch.FloatTensor(nonSeizure)
            if nonSeizure.shape != self.lastSeizure[0].shape:
                # print("uh oh")
                return self.lastSeizure
            return (nonSeizure, 0)
        else:
            seizureTimes = seizureDF['starts_sec'].tolist()
            seizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], seizureTimes[0])
            seizure = torch.FloatTensor(seizure)
            if seizure.shape != self.lastSeizure[0].shape:
                # print("uh oh")
                return self.lastSeizure
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
    sd = SIGNSDataset(PATH_TO_DATA, 'train')
    for i, (train_batch, labels_batch) in enumerate(sd):
        print(train_batch)

    # sd = SIGNSDataset(PATH_TO_DATA, 'train')
    # print(len(sd))
    # sd = SIGNSDataset(PATH_TO_DATA, 'val')
    # print(len(sd))
    # sd = SIGNSDataset(PATH_TO_DATA, 'test')
    # print(len(sd))
