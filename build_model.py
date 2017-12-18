#This python script handles building the models.

#Import the needed libraries
import os, sys
import numpy as np
import h5py
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop


model_name = ''
dataset_name = ''
it_i = 0

#Use the sys library to get variable imputs to the python script
if len(sys.argv) < 2:
	#if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python build_model.py [name_of_dataset]')
else:
	#If the correct number of arguments are given, allocate them.
    dataset_name = sys.argv[1]

print('Building model for:', dataset_name)

#Using the hp5 library to load in the datasets files.
dataset = h5py.File('datasets/' + dataset_name + '.h5' ,'r')
#Allocate the files data.
X = dataset['X'][:]
Y = dataset['Y'][:]
fftFrameSize = dataset['fftFrameSize'][()]

#Declare an array of the dimentions of each layer of the network.
layer_dims = [int(fftFrameSize/2 + 1), int(fftFrameSize/2 + 1), int(fftFrameSize/2 + 1), int(fftFrameSize/2 + 1)]

#Pring out the dimentions to the user.
for i in layer_dims:
	print('Layer:', i, 'dims')

#Builing the modle!
#Using the seiqeunecial architecture.
model = Sequential()

model.add(LSTM(
    input_dim=layer_dims[0],
    output_dim=layer_dims[1],
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    layer_dims[2],
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=layer_dims[3]))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

#Name the model with a unique name.



string_layers = ''
for layerdim in layer_dims:
	string_layers = string_layers+str(layerdim)+'_'

model_name = 'LSTM_1_'+string_layers+str(it_i)

#Create a directory for the models if it doesn't exist.
if not os.path.exists('models'):
    os.makedirs('models')

#If this model has been built before
while os.path.exists('models/' + model_name):
	#Give it a unique name and try again.
	it_i = it_i + 1
	model_name = 'LSTM_1_'+str(layer_dims[0])+'_'+str(layer_dims[1])+'_'+str(it_i)

else :
	#If the model does not exist make a directory for it.
	os.makedirs('models/' + model_name)
	model.save('models/' + model_name +'/'+ (model_name + '.h5'))
	print('Model Built:', model_name)




