import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=200, rate=8000): #TODO 5: change the nfft and rate clean folder data that you made
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate / 1) #Sample size
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        #this class also contains self._min, self._max, and self.data = (X, y), these features was added from the model.py file when modeling