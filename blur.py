# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:47:08 2018

@author: rishabh
"""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('E:/RS/ML/Machine learning tuts/Target/Part3(Computer Vision)/09)[FreeTutorials.Us] deep-learning-convolutional-neural-networks-theano-tensorflow/Code/lena.png')

plt.imshow(img)

bw=img.mean(axis=2)  
plt.imshow(bw,cmap='gray')

#create a gaussian filter
W=np.zeros((20,20))

for i in range(20):
  for j in range(20):
    dist=(i-9.5)**2+(j-9.5)**2
    W[i,j]=np.exp(-dist/50)

W/=W.sum()#normalize the kernel

#let's see what the filter looks like 
plt.imshow(W,cmap='gray')

#now convolve
out=convolve2d(bw,W)
plt.imshow(out,cmap='gray')

print(out.shape)

#we can make output the same size as input
out=convolve2d(bw,W,mode='same')
plt.imshow(out,cmap='gray')
print(out.shape)

#in color
out3=np.zeros(img.shape)
print(out3.shape)
for i in range(3):
  out3[:,:,i]=convolve2d(img[:,:,i],W,mode='same')
out3 /= out3.max()
plt.imshow(out3)