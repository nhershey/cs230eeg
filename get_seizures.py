"""Prints out a list of file names of seizures with the desired channels.
	Each line in the file is of the form "filename,i" where i is
	- -1 if the file contains no seizures
	- 0 to n to represent the ith seizure in that file
	That is, for a file with n seizures, there will be n entries of "file,i" 
	The files are output to
    - marked_files/seizures.txt
    - marked_files/nonSeizures.txt"""

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
from tqdm import tqdm

# below path goes to full data set on raiders6
PATH_TO_DATA = "../../../jdunnmon/data/EEG/eegdbs/SEC/stanford/"
SEIZURE_STRINGS = ['sz','seizure','absence','spasm']
INCLUDED_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz',
'EEG Pg1', 'EEG Pg2', 'EEG A1', 'EEG A2', 'EEG FT9', 'EEG FT10']

# gets the indices of the desired channels in a labelsObject
def getOrderedChannels(fileName, labelsObject):
	labels = list(labelsObject)
	for i in range(0, len(labels)): # needed because might come as b strings....
		labels[i] = labels[i].decode("utf")
	orderedChannels = []
	for ch in INCLUDED_CHANNELS:
		try:
			orderedChannels.append(labels.index(ch))
		except:
			print(fileName + " failed to get channel " + ch)
			return None
	return orderedChannels

def getSeizureTimes(hdf):
	annot = hdf['record-0']['edf_annotations']
	antext = [s.decode('utf-8') for s in annot['texts'][:]]
	starts100ns = [xx for xx in annot['starts_100ns'][:]]
	df = pd.DataFrame(data=antext, columns=['text'])
	df['starts100ns'] = starts100ns
	df['starts_sec'] = df['starts100ns']/10**7
	return df[df.text.str.contains('|'.join(SEIZURE_STRINGS),case=False)]

# gets all the files
def getSeizureTuples():
	fileNames =  os.listdir(PATH_TO_DATA)
	seizure_tuples = []
	nonSeizure_tuples = []
	for i in tqdm(range(len(fileNames))):
		try:
			currentFileName = PATH_TO_DATA + fileNames[i]
			hdf = h5py.File(currentFileName)
			orderedChannels = getOrderedChannels(fileNames[i], hdf['record-0']['signal_labels'])
			seizureDF = getSeizureTimes(hdf)
			if not seizureDF.empty: # i.e., it contains a seizure annotation
				seizureTimes = seizureDF['starts_sec'].tolist()
				num_seizures_in_file = 0
				for time in seizureTimes:
					seizure_tuples.append((fileNames[i], num_seizures_in_file))
					num_seizures_in_file += 1
			else:
				nonSeizure_tuples.append((fileNames[i], -1))
			hdf.close()
		except:
			print(i, " failed.")
	writeToFile(seizure_tuples, nonSeizure_tuples)

def writeToFile(seizure_tuples, nonSeizure_tuples):
	seizure_file = open('file_markers/seizures.txt', 'w+')
	for name, count in seizure_tuples:
		seizure_file.write("%s,%s\n" % (name, count))
	
	non_seizure_file = open('file_markers/nonSeizures.txt', 'w+')
	for name, count in nonSeizure_tuples:
		non_seizure_file.write("%s,%s\n" % (name, count))

if __name__ == "__main__":
    getSeizureTuples()


