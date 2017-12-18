
#This python script handles training the models.

#Import the needed libraries
import os, sys
import numpy as np
import h5py
import keras
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

#Declare the variables of the inputs
model_name = ''
dataset_name = ''
nb_epoch = ''
it_i = 0
bestWeights = ""

#Use the sys library to get variable imputs to the python script
if len(sys.argv) < 4:
    #if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python train_model.py [name_of_model] [name_of_dataset] [num_epochs]')
else:
    #If the correct number of arguments are given, allocate them.
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    nb_epoch = sys.argv[3]

#Tell the user what's going on.
print('Training Model:', model_name)
print('Training Dataset:', dataset_name)
print('Training Epochs:', nb_epoch)

#load the Keras model and data set from the specific directory of the work space using Keras an h5py
model = load_model('models/' + model_name +'/'+ model_name + '.h5')
dataset = h5py.File('datasets/' + dataset_name +'/'+ dataset_name + '.h5' ,'r')
#get the info from the files.
X = dataset['X'][:]
Y = dataset['Y'][:]
nb_epoch = int(nb_epoch)

#create the name for the new weights of we are about to make.
weights_name = model_name +'_'+dataset_name+'_'+str(it_i)

#if this model has already been trained on this dataset we automatically load the most recent set of weights and train from the best epoch.
while os.path.exists('models/' + model_name +'/'+'weights/'+weights_name + "/" ):

    #Get all of the files in this directory.
    files = os.listdir('models/' + model_name +'/'+'weights/'+weights_name + "/")

    #Always start traing from the best epoch that was last trained.
    #initialize to -1 just is case the best one was 0.
    bestEpoch = -1
    #iterate through all the files names we got from os.listdir
    for _file in (files):
    
        #Reset the wpoch we are on.
        currEpochString = ''
        
        #Get the value of the epoch of the files from the last 4 values of the files name.
        for i in range(0, 4):
            currEpochString = currEpochString + _file[(-7+i)]
        
        #Make that value and integer to be able to know how good it is.
        currEpoch = int(currEpochString)
        #If that epoch is a larger value of that the best one we got so far the loss must have improved for that epoch.
        if( currEpoch > bestEpoch):
            #Therefore this is now our best epoch
            bestEpoch = currEpoch
            #And the best weights are the files that we are currently on.
            bestWeights = _file

    #After iterating through all the files and finding the best one load the weights.
    model.load_weights('models/' + model_name + '/weights/' + weights_name+ '/' +bestWeights)

    #Tell them what we loaded.
    print('Loaded Weights:',bestWeights)
    #Because some weights already existed for this file we need to give the weights a new name, 
    #if the new name we give it also existed then the while loop will repeat until this new name does not exist meaning we are making new files.
    it_i += 1
    weights_name = model_name +'_'+dataset_name+'_'+str(it_i)

#Make the specific directory for these weights in the correct place in the work space.
if not os.path.exists('models/' + model_name +'/'+'weights/'+weights_name):
    os.makedirs('models/' + model_name +'/'+'weights/'+weights_name)


#logs include `acc` and `loss`, and optionally include `val_loss`
#We could add the loss of the epoch in to the name of the weights but then we wouldn't be able to iterate though them and find the best one as easy.
#filepath= 'models/' + model_name +'/'+'weights/'+weights_name+'/'+'weights-improvement-{epoch:04d}-{loss:.2f}.h5'
#Instead we just save a set of the weights with a unique name to that epoch with 4 digits.
filepath= 'models/' + model_name +'/'+'weights/'+weights_name+'/'+weights_name+'_{epoch:00004d}.h5'
#Create a checkpoint class to give to the training function.
#This checkpoint will be called every time the loss of the training. It will only save the weights when the loss gets better.
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#Do this
callbacks_list = [checkpoint]
#START TRAINGING.
model.fit(X, Y, nb_epoch=nb_epoch, callbacks=callbacks_list)

#If the directory does not exist for these weights make one.
if not os.path.exists('models/' + model_name +'/'+'weights/'+weights_name):
    os.makedirs('models/' + model_name +'/'+'weights/'+weights_name)

#Print finished
print('Saved Weights:',weights_name)

