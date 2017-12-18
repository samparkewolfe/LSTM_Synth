#This script created dataset out of a folder full of audios files.

#Import the needed libraries
import os, sys
import numpy as np
import librosa
import h5py
import phase_gen


training_data_path = ''
fftFrameSize = 0
hop_length = 0
sequence_length = 8
dataset_name = ''
sample_rate = 0

#Use the sys library to get variable imputs to the python script
if len(sys.argv) < 6:
    #if the wrong number of inputs was given tell the user what is supposed to be inputted
    raise ValueError('Incorrect arguments, Correct format: python create_dataset.py [path_to_raw_data] [fft_frame_size(samples)] [hop_length(samples)] [sequence_length] [dataset_name]')
else:
    #If the correct number of arguments are given, allocate them.
    training_data_path = sys.argv[1]
    fftFrameSize = int(sys.argv[2])
    hop_length = int(sys.argv[3])
    sequence_length = int(sys.argv[4])
    dataset_name = sys.argv[5] + '_' +str(fftFrameSize)+'fs_'+str(hop_length)+'hl_'+str(sequence_length)+'sl'

#Initialise the two components of the dataset with the correct size .
X = np.zeros((1, int(fftFrameSize/2)))
Y = np.zeros((1, int(fftFrameSize/2)))

#If the directory full of audiofiles the user defined exists.
if(os.path.exists(training_data_path)):
    
    #Get all the files in the directory.
    files = os.listdir(training_data_path)
    print('Number of files in folder:',len(os.listdir(training_data_path)))
    
    fft_bank = []

    #For each audio files
    for _file in files:
        #if it's not .DS_Store.
        if(_file != '.DS_Store'):
            print("Processing:",(_file))
            #Load the audio files, this also downsamples the files to 22050 regardless.
            y, sr = librosa.load(training_data_path + ('/'+_file))
            sample_rate = sr
            #Take the STFT at the user defined frame length and hoplength
            D = librosa.stft(y, n_fft=fftFrameSize, hop_length=hop_length)
            #Take the mags and phases of the STFT frames.
            magnitude, phase = librosa.magphase(D)
            #Take the transposition of the magnitudes
            magnitude_t = magnitude.T
            #For every frame of magnitudes
            for i in range(magnitude_t.shape[0]):
                #Append the frame to the fft bank.
                fft_bank.append(magnitude_t[i])
    
    #Convert the fft bank list into a numpy array.
    fft_bank = np.array(fft_bank)

    print('Synthesising Audio')
    #Generate some phases for evey new frame of predicted magnitdues using the phase gen.py library.
    phases = phase_gen.gen_phases(fft_bank.shape[0], fftFrameSize, hop_length, sample_rate)
    print('Generated New Phases', phases.shape)

    #convert all of the predicted magnitudes and generated phases back into samples with the correct hop length.
    audio = phase_gen.fft2samples(fft_bank, phases, hop_length)

    #this is to stop librosa from saving the .wavs as 64-bit floats with no one can read and normalising it here because librosa's write to wav only normalises floats.
    maxv = np.iinfo(np.int16).max
    audio_wav = (librosa.util.normalize(audio) * maxv).astype(np.int16)


    #Add one on to the user defined sequence length.
    sequence_length = sequence_length + 1


    sequences = []
    #Make a bunch of sequences of the FFT bank that are one more than the sequence length.
    for i in range(len(fft_bank) - sequence_length):
        sequences.append(fft_bank[i:i+sequence_length])

    sequences = np.array(sequences)

    #x is all the sequences with the final extra fftframe missing.
    X = sequences[:, :-1]
    #y is that missing fft frame.
    Y = sequences[:, -1]

    #N is the number of training sequences, 
    #W is the sequence length and
    #F is the number of features of each sequence.
    print(X.shape[0], X.shape[1], (fftFrameSize/2 + 1))
    #Reshape X into a way that Keras wants it...
    X = np.reshape(X, (X.shape[0], X.shape[1], int(fftFrameSize/2 + 1)))

    #Make sure we save the floats as 32-bit (64 was taking up too much space)
    X_float32 = X.astype(np.float32)
    Y_float32 = Y.astype(np.float32)

    #Make the datasets folder if it isn't there.
    if not os.path.exists('datasets'):
        os.mkdir('datasets')


    if not os.path.exists('datasets/'+dataset_name):
        os.makedirs('datasets/'+dataset_name)


    audio_name = dataset_name+'.wav'

    if not os.path.exists('datasets/'+dataset_name+'/'+audio_name):
        librosa.output.write_wav('datasets/'+dataset_name+'/'+audio_name, audio_wav, sample_rate, norm=False)

    #Make the h5py file that we are going to save all this info in.
    save_file = h5py.File('datasets/'+dataset_name+'/'+dataset_name+'.h5', 'w')
    
    #Save the dataset with some extra info that we are going to use later.
    save_file.create_dataset('X', data=X_float32)
    save_file.create_dataset('Y', data=Y_float32)
    save_file.create_dataset('fftFrameSize', data=fftFrameSize)
    save_file.create_dataset('hop_length', data=hop_length)
    save_file.create_dataset('sample_rate', data=sample_rate)
    save_file.create_dataset('number_of_sequences', data=X.shape[0])
    save_file.create_dataset('sequence_length', data=sequence_length-1)
    save_file.close()

    print("Dataset Created:",'datasets/'+dataset_name+'/'+dataset_name+'.h5')

else:
    raise ValueError("Directory Does Not Exist")


