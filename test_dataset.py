#This script will print all of the information that is saved in a h5py dataset file for this project.

#import the needed libraries
import os, sys
import numpy as np
import librosa
import h5py

#Use the sys library to get variable imputs to the python script
dataset_name = ''
if len(sys.argv) < 2:
	#if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python test_dataset.py [path_to_dataset]')
else:
	#If the correct number of arguments are given, allocate them.
    dataset_name = sys.argv[1]

dataset = h5py.File(dataset_name,'r')
X_from_save = dataset['X'][:]
Y_from_save = dataset['Y'][:]
fftFrameSize = dataset['fftFrameSize'][()]
hop_length = dataset['hop_length'][()]
sample_rate = dataset['sample_rate'][()]
sequence_length = dataset['sequence_length'][()]

print('Testing Dataset')
print("X Train Shape:", X_from_save.shape)
print("Y Train Shape:", Y_from_save.shape)
print("fftFrameSize:", fftFrameSize)
print("hop_length:", hop_length)
print("sample_rate:", sample_rate)
print("sequence_length", sequence_length)
