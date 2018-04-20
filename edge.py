# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:26:00 2018

@author: rishabh
"""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load the image
img=mpimg.imread('lena.png')

#make it B&W[black and white]
bw=img.mean(axis=2)

#Sobel Operator-approximate gradient in Y dir 
Hy=np.array([[-1,-2,-1],
            [0,0,0],
            [-1,0,1]],dtype=np.float32)

#Sobel Operator=approximate gradient in Y dir
Hx=np.array(
          [[-1,0,-1],
          [-2,0,2],
          [-1,0,1]],dtype=np.float32)

Gx=convolve2d(bw,Hx)
plt.imshow(Gx,cmap='gray')

Gy=convolve2d(bw,Hy)
plt.imshow(Gy,cmap='gray')

#Gradient magnitude
G=np.sqrt(Gx*Gx+Gy*Gy)
plt.imshow(G,cmap='gray')

# The gradient's direction
theta=np.arctan2(Gy,Gx)
plt.imshow(theta,cmap='gray')

