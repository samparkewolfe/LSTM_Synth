#This script will create the training data with new phases

#Import the needed libraries
import os, sys
import numpy as np
import h5py
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import librosa
import phase_gen

dataset_name = ''
it_i = 0

#Use the sys library to get variable imputs to the python script
if len(sys.argv) < 2:
    #if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python pred_model.py [dataset_name]')
else:
    dataset_name  = sys.argv[1]

#Load the dataset we are going to get the impulse from (this is supposed to be the same dataset that the model was trained on)
dataset = h5py.File('datasets/'+dataset_name+'.h5','r')
print('Loaded Dataset:',dataset_name)
#Get the data from the files
X = dataset['X'][:]
Y = dataset['Y'][:]
fftFrameSize = dataset['fftFrameSize'][()]
hop_length = dataset['hop_length'][()]
sample_rate = dataset['sample_rate'][()]

#The prediction sliding window will be the array that the model is made to predict new frames from.
mags = X[0]
mags = np.append(mags, Y, axis=0)
print("Dataset Y Shape: ", Y.shape)
print("Mags Shape: ", mags.shape)

print('Synthesising Audio')
#Generate some phases for evey new frame of predicted magnitdues using the phase gen.py library.
phases = phase_gen.gen_phases(mags.shape[0], fftFrameSize, hop_length, sample_rate)
print('Generated New Phases', phases.shape)

#convert all of the predicted magnitudes and generated phases back into samples with the correct hop length.
audio = phase_gen.fft2samples(mags, phases, hop_length)

#this is to stop librosa from saving the .wavs as 64-bit floats with no one can read and normalising it here because librosa's write to wav only normalises floats.
maxv = np.iinfo(np.int16).max
audio_wav = (librosa.util.normalize(audio) * maxv).astype(np.int16)

#Create a unique name and directory for the new audiofiles and write it to wav.
if not os.path.exists('datasets/phased'):
    os.makedirs('datasets/phased')

audio_name = dataset_name+'_'+'phased'+'_'+str(it_i)+'.wav'

while os.path.exists('datasets/phased/'+audio_name):
	it_i += 1
	audio_name = dataset_name+'_'+'phased'+'_'+str(it_i)+'.wav'

if not os.path.exists('datasets/phased/'+audio_name):
	librosa.output.write_wav('datasets/phased/'+audio_name, audio_wav, sample_rate, norm=False)

print('Synthesised New Audio:',  audio.shape, audio_name)





