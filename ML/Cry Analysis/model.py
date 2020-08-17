import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat(): #this function objectives: 1. 
    #if this function was called before with the same config settings, don't execute it again. Just return the same conguration settings previous output
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
        
    #variables initialization
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    
    n_samples = 2 * int(df['length'].sum() / (config.step / config.rate)) #number of samples per all records (total records length / sampling_rate)
    prob_dist = class_dist / class_dist.sum() #the percentage of each class among the total data, assuming that you have equal number of records from each class
    for _ in tqdm(range(n_samples)):
        #getting a random sample
        rand_class = np.random.choice(class_dist.index, p=prob_dist) #returns a random class name based on the probability distribution of each class
        file = np.random.choice(df[df.label == rand_class].index) #returns a random wav file name of the rand_class
        rate, wav = wavfile.read('wavfiles/' + file) #read the wav file (MOD 1: wavfiles => clean)
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index: rand_index + config.step] #returns a random window(chunk) from the wav file
        
        #convert from the time domain into mfcc
        X_sample = mfcc(sample, rate, numcep = config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        
        #Store the values locally
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(rand_class))
    
    #Normalization
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    
    #reshaping to fit each NN type
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=len(classes))
    
    #store the values globally
    config.min = _min
    config.max = _max
    config.data = (X, y)
    config.classes = classes
    #store the configuration settings and output in a file to prevent the rerun of this method with the same configuration settings
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    return X, y

def get_conv_model():
    model = Sequential() #initializing the model with now hidden layers
    #add layer after another using add
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
              padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam'
                  , metrics=['acc'])
    return model

def get_recurrent_model():
        #shape of data for RNN is (n, time, feat)
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(TimeDistributed(Dense(32, activation='relu')))
        model.add(TimeDistributed(Dense(16, activation='relu')))
        model.add(TimeDistributed(Dense(8, activation='relu')))
        model.add(Flatten())
        model.add(Dense(len(classes), activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam'
                      , metrics=['acc'])
        return model
        
#reading data in
df = pd.read_csv('Cries.csv') #TODO 1: change the reference to ur csv
df.set_index('fname', inplace=True)

#adding the length of each wav file in the df
for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label)) #classes = class names ([burping, hungry,...])
class_dist = df.groupby(['label'])['length'].mean() #average length of each class wav files

#print the class distribution
'''
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
'''

#initialize variables
config = Config(mode='conv') #basic variables
X, y = build_rand_feat() #return random samples from the classes to prevent class inbalance, and return the mfcc version not the time domain


if config.mode == 'conv':
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
elif config.mode == 'time':
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()


y_flat = np.argmax(y, axis=1) #return the index corresponding to the 1 value in the y array
classWeight = compute_class_weight('balanced', np.unique(y_flat), y_flat) #A way to modify the hyperparameters a little bit to reduce the bias and improve accuracy a little bit
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1
                             , mode='max', save_best_only=True, save_weights_only=False, period=1) #this line makes us update the model we saved before only if there is an acc improvement

#build and save the model
model.fit(X, y, epochs=10, batch_size=32, shuffle=True, class_weight=classWeight, validation_split=0.1) #validation split lets the fit function test the model against 0.1(10%) of the data
model.save(config.model_path)

#TODO: All of that must be after you make the number of records equal in each category
#TODO: find another way to handle class imbalance other than this way
#TODO: change the samples time, to best fit the problem
#TODO: remove the randomness during choosing the training data, and make samples scan all the records.
#TODO: to increase performance u can change the number of samples, NN archeticture