# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:02:27 2018

@author: rishabh
"""

import matplotlib.pyplot as plt
import numpy as np
import wave

from scipy.io.wavfile import write

spf=wave.open('E:/RS/ML/Machine learning tuts/Target/Part3(Computer Vision)/09)[FreeTutorials.Us] deep-learning-convolutional-neural-networks-theano-tensorflow/Code/helloworld.wav','r')

signal=spf.readframes(-1)
signal=np.fromstring(signal,'Int16')
print("numpy signal",signal.shape)

plt.plot(signal)
plt.title("Hello World without echo")

delta=np.array([1,0,0])
noecho=np.convolve(signal,delta)
print("noecho signal",noecho.shape)

filt=np.zeros(16000)
filt[0]=1
filt[4000]=0.6
filt[8000]=0.3
filt[12000]=0.2
filt[15999]=0.1
out=np.convolve(signal,filt)

out.astype(np.int16)
write('E:/RS/ML/Machine learning tuts/Target/Part3(Computer Vision)/09)[FreeTutorials.Us] deep-learning-convolutional-neural-networks-theano-tensorflow/Code/out.wav',16000,out)

plt.plot(out)
plt.title("Hello World with echo")
