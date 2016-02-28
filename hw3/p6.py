import numpy as np
from scipy import io as sio
from numpy import linalg as la
from matplotlib import pyplot as plt

HOUSING_DATA_PATH = 'data/housing_dataset/'
FEATURES = 8

def add_dim(x):
  ones = np.array([[1]*x.shape[0]])
  xnew = np.concatenate((x,ones.T),1)
  assert(xnew.shape[0] == x.shape[0])
  assert(xnew.shape[1] == x.shape[1] + 1)
  return xnew

def load_data_housing():
  mat = sio.loadmat(HOUSING_DATA_PATH + 'housing_data.mat')
  xtrain = mat['Xtrain']
  ytrain = mat['Ytrain']
  xtest = mat['Xvalidate']
  ytest = mat['Yvalidate']
  xtrain = add_dim(xtrain)
  xtest = add_dim(xtest)
  return xtrain,ytrain,xtest,ytest

def find_weights(x,y):
  xtx = np.dot(np.transpose(x),x)
  xty = np.dot(np.transpose(x),y)
  # w = np.dot(la.inv(xtx),xty)
  w = la.solve(xtx,xty)
  # w,resid = la.lstsq(x,y)[:2]
  return w

def classify(x,w):
  return np.dot(x,w)

def error(yp,y):
  return la.norm(yp-y)**2

def main():
  x,y,xt,yt = load_data_housing()
  w = find_weights(x,y)
  yp = classify(x,w)
  rss = error(yp,y)

  ytp = classify(xt,w)
  rsst = error(ytp,yt)
  print rsst

  print min(ytp),max(ytp)

  plt.figure()
  plt.plot(xrange(FEATURES),[w[i] for i in xrange(FEATURES)],marker='o')
  plt.savefig('out/6C.png')

  plt.figure()
  plt.hist(yp-y,bins=69) # lol
  plt.savefig('out/6D.png')

if __name__=='__main__':
  main()
