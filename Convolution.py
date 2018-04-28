# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:43:46 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from datetime import datetime

def convolve2d(X,W):
  t0=datetime.now()
  n1,n2=X.shape
  m1,m2=W.shape
  Y=np.zeros((n1+m1-1,n2+m2-1))
  for i in range(n1+m1-1):
    for ii in range(m1):
      for j in range(n2+m2-1):
        for jj in range(m2):
          if i>ii and j>jj and i-ii<n1 and j-jj:
            Y[i,j]+=W[ii,jj]*X[i-ii,j-jj]

  