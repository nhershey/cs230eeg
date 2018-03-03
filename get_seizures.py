from __future__ import print_function, division, unicode_literals

import pandas as pd
import numpy as np
import h5py
from pprint import pprint

import os
import sys
import pandas as pd
from collections import Counter
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

# below path goes to full data set on raiders6
#PATH_TO_DATA = "../../../../../jdunnmon/EEG/eegdbs/SEC/stanford/"
PATH_TO_DATA = "data/dummy_data/"
SEIZURE_STRINGS = ['sz','seizure','absence','spasm']
FREQUENCY = 200
EPOCH_LENGTH_SEC = 10
INCLUDED_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz',
'EEG Pg1', 'EEG Pg2', 'EEG A1', 'EEG A2', 'EEG FT9', 'EEG FT10']

random.seed(1)

# gets the indices of the desired channels in a labelsObject
def getOrderedChannels(labelsObject):
	labels = list(labelsObject)
	for i in range(0, len(labels)): # needed because might come as b strings....
		labels[i] = labels[i].decode("utf")
	orderedChannels = []
	for ch in INCLUDED_CHANNELS:
		try:
			orderedChannels.append(labels.index(ch))
		except:
			print("failed to get channel " + ch)
			return None
	return orderedChannels

# slices a signals matrix at sliceTime
def sliceEpoch(orderedChannels, signals, sliceTime):
	startTime = int(FREQUENCY * sliceTime)
	endTime = int(FREQUENCY * (sliceTime + EPOCH_LENGTH_SEC))
	sliceMatrix = signals[orderedChannels,  startTime : endTime]
	# return sliceMatrix
	sliceVector = np.ndarray.flatten(sliceMatrix)
	return sliceVector

# gets a random slice from a record
def getRandomSlice(record):
	orderedChannels = getOrderedChannels(record['signal_labels'])
	if orderedChannels == None:
		return None
	maxStart = record['signals'].shape[1] - FREQUENCY * EPOCH_LENGTH_SEC
	return sliceEpoch(orderedChannels, record['signals'], random.randint(0,maxStart))

# gets all the files
def getDataLoaders():
	fileNames =  os.listdir(PATH_TO_DATA)
	filesWithoutSeizures = []
	seizures = []
	nonSeizures = []
	for i in range(len(fileNames)):
		currentFileName = PATH_TO_DATA + fileNames[i]
		hdf = h5py.File(currentFileName)
		annot = hdf['record-0']['edf_annotations']
		antext = [s.decode('utf-8') for s in annot['texts'][:]]
		starts100ns = [xx for xx in annot['starts_100ns'][:]]
		df = pd.DataFrame(data=antext, columns=['text'])
		df['starts100ns'] = starts100ns
		df['starts_sec'] = df['starts100ns']/10**7
		seizureDF = df[df.text.str.contains('|'.join(SEIZURE_STRINGS),case=False)]
		if not seizureDF.empty: # i.e., it contains a seizure annotation
			seizureTimes = seizureDF['starts_sec'].tolist()
			orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])
			for time in seizureTimes:
				seizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], time)
				if seizure is not None:
					seizures.append((torch.FloatTensor(seizure),1))
		else: # if no seizure, we randomly pull 10 seconds and call it non-seizure
			filesWithoutSeizures.append(fileNames[i])
			orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])
			maxStart = float(hdf['record-0']['signals'].shape[1] - FREQUENCY * EPOCH_LENGTH_SEC)
			nonSeizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], random.randint(0,1000.0))
			nonSeizures.append((torch.FloatTensor(nonSeizure),0))
		hdf.close()
	allData = seizures + nonSeizures
	return allData
