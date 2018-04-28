# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 02:46:41 2018

@author: rishabh
"""


import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import scipy.io


from sklearn.utils import shuffle

from datetime import datetime

def errorRate(y,t):
  return np.mean(y!=t)
  
def relu(a):
  return a*(a>0)
  
def y2indicator(y):
  N=len(y)
  K=len(set(y))
  ind=np.zeros((N,K))
  for i in range(N):
    ind[i,y[i]]=1
  return ind

def convpool(X,W,b,poolsize=(2,2)):
  conv_out=conv2d(input=X,filters=W)
  
  pooled_out=pool.pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True)
  return relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))#not understood


def init_filter(shape,poolsz):#not undertood
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[1:]))#not understood
    return w.astype(np.float32)
  
def rearrange(X):
  '''
  N=X.shape[-1]
  out=np.zeros((N,3,32,32),dtype=np.float32)
  for i in range(N):
    for j in range(3):
      out[i,j,:,:]=X[:,:,j,i]
  '''
  return (X.transpose(3,2,0,1)/255).astype(np.float32)
  

def main():
  # Need to scale! don't leave as 0..255
  # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
  # So flatten it and make it 0..9
  # Also need indicator matrix for cost calculation
 
  # step 1: load the data, transform as needed
  train=scipy.io.loadmat('C:/Users/rishabh/Desktop/train_32x32.mat')
  test=scipy.io.loadmat('C:/Users/rishabh/Desktop/test_32x32.mat') 
  
  Xtrain=rearrange(train['X'])
  Ytrain=train['y'].flatten()-1#not undertood
  del train#not undertood
  Xtrain,Ytrain=shuffle(Xtrain,Ytrain) 
  Ytrain_ind=y2indicator(Ytrain)
  
  Xtest=rearrange(test['X'])
  Ytest=test['y'].flatten()-1#not undertood
  del test#not undertood
  Ytest_ind=y2indicator(Ytest)
  
  max_iter=6
  print_period=10
  
  N=Xtrain.shape[0]
  lr=np.float32(1e-2)
  mu=np.float32(0.99)
  reg=np.float32(0.01)
  batch_sz=500
  n_batches=N//batch_sz
  
  M=500
  K=10
  poolz=(2,2)
  
  # after conv will be of dimension 32 - 5 + 1 = 28
  # after downsample 28 / 2 = 14
  W1_shape=(20,3,5,5)
  W1_init=init_filter(W1_shape,poolz)
  b1_init=np.zeros(W1_shape[0],dtype=np.float32)
  
  # after conv will be of dimension 14 - 5 + 1 = 10
  # after downsample 10 / 2 = 5
  W2_shape=(50,20,5,5)
  W2_init=init_filter(W2_shape,poolz)
  b2_init=np.zeros(W2_shape[0],dtype=np.float32)
  
  #ANN weights
  W3_init=np.random.randn(W2_shape[0]*5*5,M)/np.sqrt(W2_shape[0]*5*5+M)
  b3_init=np.zeros(M,dtype=np.float32)
  W4_init=np.random.randn(M,K)/np.sqrt(M+K)
  b4_init=np.zeros(K,dtype=np.float32)
  
  # step 2: define theano variables and expressions
  X=T.tensor4('X',dtype='float32')
  Y=T.ivector('T')
  W1=theano.shared(W1_init.astype(np.float32) ,'W1')
  b1=theano.shared(b1_init,'b1')
  W2=theano.shared(W2_init.astype(np.float32),'W2')
  b2=theano.shared(b2_init,'b2')
  W3=theano.shared(W3_init.astype(np.float32),'W3')
  b3=theano.shared(b3_init,'b3')
  W4=theano.shared(W4_init.astype(np.float32),'W4')
  b4=theano.shared(b4_init,'b4')
  
  #forward pass
  Z1=convpool(X,W1,b1)
  Z2=convpool(Z1,W2,b2)
  Z3=relu(Z2.flatten(ndim=2).dot(W3)+b3)
  pY=T.nnet.softmax(Z3.dot(W4)+b4)
  
  # define the cost function and prediction
  cost = -(T.log(pY[T.arange(Y.shape[0]), Y])).mean()#not understood
  prediction = T.argmax(pY, axis=1)
  
  #training expressions and functions
  params=[W1,b1,W2,b2,W3,b3,W4,b4]

  #moment changes
  dparams=[
    theano.shared(np.zeros_like(
                p.get_value(),dtype=np.float32) 
        )for p in params         
  ]
  
  updates=[]  
  grads=T.grad(cost,params)
  for p,dp,g in zip(params,dparams,grads):
    dp_update=mu*dp-lr*g
    p_update=p+dp_update
    
    updates.append((dp,dp_update))
    updates.append((p, p_update))
    
  train=theano.function(
                inputs=[X,Y],
            updates=updates,)
  
  get_prediction=theano.function(
                inputs=[X,Y],
                outputs=[cost,prediction])
  
  
  t0=datetime.now()
  costs=[]
  for i in range(max_iter):
    Xtrain,Ytrain=shuffle(Xtrain,Ytrain)
    for j in range(n_batches):
      batchX=Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
      batchY=Ytrain[j*batch_sz:(j*batch_sz+batch_sz),]
          
      train(batchX,batchY)
      if j%print_period==0:
        cost_val,prediction_val=get_prediction(Xtest,Ytest)
        err=errorRate(prediction_val,Ytest)
        print("Cost / err at iteration i=%d j%d: %.3f/%.3f"%(i,j,cost_val,err))
        costs.append(cost_val)
  print("Elasped time",(datetime.now()-t0))
  plt.plot(costs)
  
  
if __name__ == '__main__':
    main()  