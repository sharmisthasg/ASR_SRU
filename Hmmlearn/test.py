# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:36:21 2020

@author: Aditya Aggarwal
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import wavfile
import librosa
from pydub import AudioSegment
import speech_recognition as sr 
from hmmlearn.hmm import GaussianHMM
from librosa.feature import mfcc
f,a = wavfile.read('./1.wav')
a1 = a[:f*10].astype(np.float)
a2 = a[f*10:f*20].astype(np.float)
f1 = librosa.feature.mfcc(a1 ,n_mfcc=39)
f2 = librosa.feature.mfcc(a2 ,n_mfcc=39)
print(f1,f2)
mhmm = GaussianHMM(n_components=2)
mhmm.fit(f1)
x = mhmm.predict(f2)
print(x)




























































































































































































