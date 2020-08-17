import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score


def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0] - config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat
                     , nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min / (config.max - config.min))
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, fn_prob
    
df = pd.read_csv('Cries.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))

p_path = os.path.join('pickles', 'conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean')
acc_score = accuracy_score(y_true, y_pred)

'''
record_y_pred = []
record_y_true = []
for fn in fn_prob:
    record_y_true.append(classes.index(fn2class[fn]))
    record_y_pred.append(np.argmax(fn_prob[fn]))

record_acc_score = accuracy_score(record_y_true, record_y_pred)
'''

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    #for c, p in zip(classes, y_prob):
        #df.at[i, c] = p
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

#df.to_csv('prediction.csv', index=False)

#showing the percentage of records
'''
cnt = df.groupby('label').count()
total = sum(cnt.iloc[:, 0])
cnt.iloc[:, 1] = [n/total for n in cnt.iloc[:, 0]]
print(cnt)
'''


'''
for fn in tqdm(os.listdir(audio_dir)):
        #audio_dir = 'clean'
        #fn = '549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav'
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0] - config.step, config.step):
            #i = 0
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat
                     , nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min / (config.max - config.min))
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
        y_pred.append(np.argmax(fn_prob[fn]))
        y_true.append(c)
        
    return y_true, y_pred, fn_prob
'''













