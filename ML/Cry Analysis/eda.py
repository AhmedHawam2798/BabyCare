import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def plot_signals(signals, n_rows, n_cols):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False,
                             sharey=True, figsize=(20,2.5 * n_rows))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(n_rows):
        for y in range(n_cols):
            axes[y].set_title(list(signals.keys())[i]) #TODO 4.1: change the index to [x, y] if the rows are gt 1
            axes[y].plot(list(signals.values())[i])
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft, n_rows, n_cols):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False,
                             sharey=True, figsize=(20,2.5 * n_rows))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(n_rows):
        for y in range(n_cols):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[y].set_title(list(fft.keys())[i])
            axes[y].plot(freq, Y)
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank, n_rows, n_cols):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False,
                             sharey=True, figsize=(20,2.5 * n_rows))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(n_rows):
        for y in range(n_cols):
            axes[y].set_title(list(fbank.keys())[i])
            axes[y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs, n_rows, n_cols):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False,
                             sharey=True, figsize=(20,2.5 * n_rows))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(n_rows):
        for y in range(n_cols):
            axes[y].set_title(list(mfccs.keys())[i])
            axes[y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return [Y, freq]
    
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


df = pd.read_csv('Cries.csv') #TODO 1: set the reference to ur csv file name(the file contains two columns fname and label)
df.set_index('fname', inplace=True)
for f in df.index:
    rate, signal = wavfile.read('wavfiles/' + f)
    df.loc[f, 'length'] = signal.shape[0]/rate
#print(df.head())
#rate is equal 8000

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()
#print(class_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels = class_dist.index, autopct='%1.1f%%')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[3, 0]
    signal, rate = librosa.load('wavfiles/' + wav_file, sr=8000) #TODO 2: change the sr(sampling rate) with the rate you've got from the previous loop rate variable
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=200).T #TODO 3: change the nfft = rate / 40(the usual optimal window_length)
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=200).T #TODO 3 again
    mfccs[c] = mel

plot_signals(signals, 1, 5) #TODO 4.2: change the n_rows, n_cols (ex: 2*5)
plt.show()

plot_fft(fft, 1, 5)
plt.show()

plot_fbank(fbank, 1, 5)
plt.show()

plot_mfccs(mfccs, 1, 5)
plt.show()

#Down Sampling and removing dead parts in all records
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/' + f, sr=8000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/' + f, rate=rate, data=signal[mask])


'''
team meeting agenda
1. khaled problem and how we solved it
2. dellemc bi-weekly meeting and how to benefit from them and how to interact with them
3. taking a week off
'''























