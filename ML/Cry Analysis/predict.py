# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:16:06 2020

@author: DiaaAbdElmaksoud@hotmail.com
"""
import pickle
import os
import numpy as np
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
import librosa


def predict_path(wav_path):
    '''
    predict_path method returns the prediction of the baby's crying reason
    input: the full path of the wav audio file 
    output: a category of 5
    '''
    
    wav, rate = librosa.load(wav_path, 8000)
    y_prob = []
    
    for i in range(0, wav.shape[0] - config.step, config.step):
        sample = wav[i:i+config.step]
        
        x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        x = (x - config.min / (config.max - config.min))
        if config.mode == 'conv':
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
        elif config.mode == 'time':
            x = np.expand_dims(x, axis=0)
            
        y_hat = model.predict(x)
        y_prob.append(y_hat)
    
    fn_prob = np.mean(y_prob, axis=0).flatten()
    y_pred = np.argmax(fn_prob)
    classes = config.classes
    
    return classes[y_pred]
    

# Load model and config
p_path = os.path.join('pickles', 'conv.p')
model_path = os.path.join('models', 'conv.model')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
model = load_model(model_path)


# Testing the predict_path method
'''
audio_dir = 'clean'
fn = 'f5b29377-7cd6-4688-942c-5a07add39dc5-1437480263225-1.7-f-26-dc.wav'
wav_path = os.path.join(audio_dir, fn)
print(predict_path(wav_path))
'''

#TODO: in a record 6.2 seconds, the yhat comes out with very strange large numbers at the 6th index, look at screenshot 748







