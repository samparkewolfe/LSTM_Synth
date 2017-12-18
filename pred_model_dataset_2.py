#This script deals with predicting a new audiofile from the neural network.

#Import the needed libraries
import os, sys
import numpy as np
import h5py
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import librosa
import phase_gen
import time
import math

model_name = ''
weights_name_1 = ''
epoch_1 = ''
weights_name_2 = ''
epoch_2 = ''
dataset_name = ''
pred_length = ''
dataset_iteration = ''

it_i = 0

#Use the sys library to get variable imputs to the python script
if len(sys.argv) < 9:
    #if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python pred_model.py [name_of_model] [weights_name_1] [epoch_1(0000)] [weights_name_2] [epoch_2(0000)] [name_of_dataset] [dataset_iteration/rand] [pred_length(frames)]')
else:
    #If the correct number of arguments are given, allocate them.
    model_name = sys.argv[1]
    weights_name_1 = sys.argv[2]
    epoch_1 = sys.argv[3]
    weights_name_2 = sys.argv[4]
    epoch_2 = sys.argv[5]
    dataset_name  = sys.argv[6]
    dataset_iteration = sys.argv[7]
    pred_length = sys.argv[8]

#Load the model that the user wants to predict from.
print('Predicting From Model')
model_1 = load_model('models/' + model_name +'/'+ model_name + '.h5')
model_2 = load_model('models/' + model_name +'/'+ model_name + '.h5')
model_3 = load_model('models/' + model_name +'/'+ model_name + '.h5')
print('Loaded Model:', model_name)


bestWeights_1 = ''
#If the models weights exists.
if(os.path.exists('models/' + model_name +'/'+'weights/'+weights_name_1 + "/" )):

    #Find the files inside the weights of the model for the user 
    files = os.listdir('models/' + model_name +'/'+'weights/'+weights_name_1 + "/")

    print('Loading weights: ', weights_name_1)

    #If the user wants to predict from the best epoch of these weights then get the best epoch.
    if(epoch_1 == 'best'):
        print("Finding Best Epoch")
        bestEpoch = -1
        for _file in (files):
            if(_file[-3:] == '.h5'):
                print(_file, _file[-3:])
                currEpochString = ''
                for i in range(0, 4):
                    currEpochString = currEpochString + _file[(-7+i)]
                print(currEpochString)
       	        currEpoch = int(currEpochString)
                if( currEpoch > bestEpoch):
                    bestEpoch = currEpoch
                    bestWeights_1 = _file[:-3]
    else:
        bestWeights_1 = weights_name_1+'_'+epoch_1


#Load the chosen weights
model_1.load_weights('models/' + model_name +'/'+'weights'+'/'+ weights_name_1+'/'+bestWeights_1+'.h5')
model_3.load_weights('models/' + model_name +'/'+'weights'+'/'+ weights_name_1+'/'+bestWeights_1+'.h5')
print('Loaded Weights 1:',bestWeights_1)


bestWeights_2 = ''
#If the models weights exists.
if(os.path.exists('models/' + model_name +'/'+'weights/'+weights_name_2 + "/" )):
    
    #Find the files inside the weights of the model for the user 
    files = os.listdir('models/' + model_name +'/'+'weights/'+weights_name_2 + "/")

    #If the user wants to predict from the best epoch of these weights then get the best epoch.
    if(epoch_2 == 'best'):
        bestEpoch = -1
        for _file in (files):
            if(_file[-3:] == '.h5'):
                print(_file, _file[-3:])
                currEpochString = ''
                for i in range(0, 4):
                    currEpochString = currEpochString + _file[(-7+i)]
                print(currEpochString)
                currEpoch = int(currEpochString)
                if( currEpoch > bestEpoch):
                    bestEpoch = currEpoch
                    bestWeights_2 = _file[:-3]
    else:
        bestWeights_2 = weights_name_2+'_'+epoch_2

#Load the chosen weights
model_2.load_weights('models/' + model_name +'/'+'weights'+'/'+ weights_name_2+'/'+bestWeights_2+'.h5')
print('Loaded Weights 2:',bestWeights_2)


# print('Weight Arrays')
# for layer_it in range (len(model_1.layers)):
#     new_weights = []
#     print("original len:", len(model_1.layers[layer_it].get_weights()))

#     for weightarray_it in range (len(model_1.layers[layer_it].get_weights())):
#         weight_array_1 = model_1.layers[layer_it].get_weights()[weightarray_it]
#         weight_array_2 = model_2.layers[layer_it].get_weights()[weightarray_it]
#         print(weight_array_1.shape)
#         new_weights.append(weight_array_1)

#     print("new len:", len(new_weights))
#     model_1.layers[layer_it].set_weights(new_weights)



pred_length = int(pred_length)

#Load the dataset we are going to get the impulse from (this is supposed to be the same dataset that the model was trained on)
dataset = h5py.File('datasets/'+dataset_name+'/'+dataset_name+'.h5','r')
print('Loaded Dataset:',dataset_name)
#Get the data from the files
X = dataset['X'][:]
fftFrameSize = dataset['fftFrameSize'][()]
hop_length = dataset['hop_length'][()]
sample_rate = dataset['sample_rate'][()]

#If the user wants a random sequence from the dataset to use as an impulse
if(dataset_iteration == 'rand'):
    dataset_iteration = np.random.randint(X.shape[0]-1)
else:
    #Else the user can define a frame.
    dataset_iteration = int(dataset_iteration)

impulse_it = dataset_iteration
print(impulse_it)

print('Predicting', pred_length, 'frames...')


#The prediction sliding window will be the array that the model is made to predict new frames from.
pred_sliding_window = X[impulse_it]
pred_mags = []

#Fill the predicted magnitudes with the impulse so the user can hear it in their predicted audio.
for i in range(X[impulse_it].shape[0]):
    pred_mags.append(X[impulse_it][i])

#Inorder to predict from the keras model it needs to be in this strange shape.
pred_sliding_window = np.reshape(pred_sliding_window, (1, pred_sliding_window.shape[0], pred_sliding_window.shape[1]))

interpolation_amount = 0
interpolation_increment = 1/pred_length

startTime = time.clock()
#For every frame the user want to predict.
for i in range (pred_length):

    # print('Weight Arrays')
    for layer_it in range (len(model_1.layers)):
        new_weights = []
        # print("original len:", len(model_1.layers[layer_it].get_weights()))

        for weightarray_it in range (len(model_1.layers[layer_it].get_weights())):
            weight_array_1 = model_1.layers[layer_it].get_weights()[weightarray_it]
            weight_array_2 = model_2.layers[layer_it].get_weights()[weightarray_it]
            # print(weight_array_1.shape)
            new_weights.append( ( weight_array_1*( 1-((math.sin(2*math.pi*interpolation_amount)+1)/2.0) ) ) + ( weight_array_2*( (math.sin(2*math.pi*interpolation_amount)+1)/2.0 ) ) )

        # print("new len:", len(new_weights))
        model_3.layers[layer_it].set_weights(new_weights)

    print ((math.sin(2*math.pi*interpolation_amount)+1)/2.0 )
    interpolation_amount += interpolation_increment
    #Predict a new frame with the sequence stored in the sliding window.
    pred_mag = model_3.predict(pred_sliding_window)

    #Append the new frame to the pred mags array
    pred_mags.append(pred_mag[0])

    #Make the sequence to predict the next frame from pred mags as well.
    temp_pred_sliding_window = np.array(pred_mags[(i+1):])

    #Reshape pred sliding window in to the weird shape for keras.
    pred_sliding_window = np.reshape(temp_pred_sliding_window, (1, temp_pred_sliding_window.shape[0], temp_pred_sliding_window.shape[1]))


pred_mags = np.array(pred_mags)

print('Finished Predicting', pred_mags.shape, pred_mags.min(), pred_mags.max())

print('Synthesising Audio')
#Generate some phases for evey new frame of predicted magnitdues using the phase gen.py library.
phases = phase_gen.gen_phases(pred_mags.shape[0], fftFrameSize, hop_length, sample_rate)
print('Generated New Phases', phases.shape)

#convert all of the predicted magnitudes and generated phases back into samples with the correct hop length.
audio = phase_gen.fft2samples(pred_mags, phases, hop_length)

#this is to stop librosa from saving the .wavs as 64-bit floats with no one can read and normalising it here because librosa's write to wav only normalises floats.
maxv = np.iinfo(np.int16).max
audio_wav = (librosa.util.normalize(audio) * maxv).astype(np.int16)

endTime = time.clock()

print("Total Time: " + str(endTime-startTime))
print("Time Per Prediction: " + str((endTime-startTime)/pred_length))

#Create a unique name and directory for the new audiofiles and write it to wav.
if not os.path.exists('models/' + model_name +'/'+'weights/'+weights_name_1+'/synthesised_audio'):
    os.makedirs('models/' + model_name +'/'+'weights/'+weights_name_1+'/synthesised_audio')

audio_name = bestWeights_1 + '_' + str(impulse_it)+'impls' + '_' + str(it_i) + '.wav'

while os.path.exists('models/' + model_name +'/'+'weights/'+weights_name_1+'/synthesised_audio/'+audio_name):
	it_i += 1
	audio_name = bestWeights_1 + '_' + str(impulse_it)+'impls' + '_' + str(it_i) + '.wav'

if not os.path.exists('models/' + model_name +'/'+'weights/'+weights_name_1+'/synthesised_audio/'+audio_name):
	librosa.output.write_wav('models/' + model_name +'/'+'weights/'+weights_name_1+'/synthesised_audio/'+audio_name, audio_wav, sample_rate, norm=False)

print('Synthesised New Audio:',  audio.shape, audio_name)





