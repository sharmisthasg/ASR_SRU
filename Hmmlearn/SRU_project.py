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
import hmmlearn as hmm

data = pd.read_csv('train.tsv',sep="\t")
speech_path = data['path']    

freq = []
audio = []
for i in range(len(speech_path)):
    sound = AudioSegment.from_mp3('./clips/' + speech_path[i])
    speech_path[i] = speech_path[i].replace('.mp3','.wav')
    sound.export('./wavfiles/' + speech_path[i], format="wav")
    
for i in range(len(speech_path)):
    fs,data = wavfile.read('./wavfiles/' + speech_path[i])
    freq.append(fs)
    audio.append(data)
    

r = sr.Recognizer() 
f,audio1 = wavfile.read('./1.wav')
audio1 = (np.iinfo(np.int32).max * (audio1/np.abs(data).max())).astype(np.int32)
wavfile.write('./test.wav', f, audio1)

with sr.WavFile('./test.wav') as source:
    audio1 = r.record(source)
try: 
    print("The audio file contains: " + r.recognize_google(audio1)) 
except sr.UnknownValueError: 
    print("Google Speech Recognition could not understand audio") 
except sr.RequestError as e: 
    print("Could not request results from Google Speech Recognition service; {0}".format(e)) 

