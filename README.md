# Final Research Project: LSTM Synth

This project involved training recursive neural networks to predict audio sequences in the form of STFT magnitudes.

In python, a neural network is trained on the sequences of the STFT transform of an audio file. Then for synthesis the network is made to recursively predict STFT Mag frames from the last STFT Mags it’s just predicted.

After an initial impulse of known magnitudes is given to the network to get it started, the network eventually is only predicting frames from it’s own predictions.

[Listen to some example sounds here](https://soundcloud.com/user-106787896/sets)