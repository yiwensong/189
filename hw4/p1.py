import numpy as np
from scipy import io as sio
from numpy import linalg as la
from matplotlib import pyplot as plt

import random

HOUSING_DATA_PATH = 'housing_dataset/'
FEATURES = 8

def add_dim(x):
  ones = np.array([[1]*x.shape[0]])
  xnew = np.concatenate((x,ones.T),1)
  assert(xnew.shape[0] == x.shape[0])
  assert(xnew.shape[1] == x.shape[1] + 1)
  return xnew

def sub_mean(X,y):
  xbar = np.array(map(np.average,X))
  xbar = np.array([xbar]*len(X[0])).T
  ybar = np.average(y)
  X2 = X - xbar
  y2 = np.array(y)-ybar
  return X2,y2

def load_data_housing():
  mat = sio.loadmat(HOUSING_DATA_PATH + 'housing_data.mat')
  xtrain = mat['Xtrain']
  ytrain = mat['Ytrain']
  xtest = mat['Xvalidate']
  ytest = mat['Yvalidate']
  # xtrain = add_dim(xtrain)
  # xtest = add_dim(xtest)
  return xtrain,ytrain,xtest,ytest

def find_weights(x,y,l=0):
  x2,y2 = sub_mean(x,y)
  xtx = np.dot(np.transpose(x2),x2) + np.diag([l]*len(x2[0]))
  xty = np.dot(np.transpose(x2),y2)
  # w = np.dot(la.inv(xtx),xty)
  w = la.solve(xtx,xty)
  # w,resid = la.lstsq(x,y)[:2]
  return w

def classify(x,w,y):
  x,u = sub_mean(x,y)
  return np.dot(x,w) + np.average(y)

def error(yp,y):
  return la.norm(yp-y)**2

def k_folds(k,size):
  indices = range(size)
  random.shuffle(indices)
  fold_size = size/k
  folds = [0] * k
  for i in xrange(k):
    folds[i] = indices[i*fold_size:(i+1)*fold_size]
  return folds

def find_opt_lambda(X,y):
  NUM_FOLDS = 10
  l_vals = np.logspace(-10,10,50)
  best = None
  for lam in l_vals:
    print 'lambda',lam
    folds = k_folds(NUM_FOLDS,len(X))
    succ = range(NUM_FOLDS)
    for i in xrange(NUM_FOLDS):
      v_sel = folds[i]
      v_X = [X[idx] for idx in v_sel]
      v_y = [y[idx] for idx in v_sel]

      t_sel = np.concatenate((np.ndarray.flatten(np.array(folds[0:i],dtype='int')),\
          np.ndarray.flatten(np.array(folds[i+1:],dtype='int'))))
      t_X = [X[idx] for idx in t_sel]
      t_y = [y[idx] for idx in t_sel]

      w = find_weights(t_X,t_y,lam)
      p_v_y = classify(v_X,w,t_y)
      succ[i] = -np.dot((p_v_y - v_y).T,(p_v_y - v_y))
    succ = np.average(succ)
    print 'squared error',succ
    print ''
    if best is None or succ > best:
      best = succ
      best_lam = lam

  return best_lam

def main():
  x,y,xt,yt = load_data_housing()
  lam = find_opt_lambda(x,y)
  w = find_weights(x,y,lam)
  pyt = classify(xt,w,y)
  sqerr = np.dot((yt - pyt).T,(yt-pyt))
  print sqerr
  plt.plot(w)
  plt.savefig('out/p1.png')

if __name__=='__main__':
  main()
