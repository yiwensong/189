import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy import io as sio
from scipy import stats
import random
import pandas as pd

from p2 import *

DIGIT_PATH = 'data/digit_dataset/'
SPAM_PATH = 'spam_dataset/'
NUM_CLASSES = 10
EP = .00001

def format_img(img):
  img = np.swapaxes(img,0,2)
  img = img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
  return img

def load_data_digits():
  ''' Loads the digit dataset according to the shitty format
  that we are given'''
  # In this one, the test_img are in format of (10000,784)
  # Fucking inconsistent as shit
  test_img = sio.loadmat(DIGIT_PATH + 'test.mat')['test_images']

  # Unfortunately, the training set is still fucked up
  train_img = sio.loadmat(DIGIT_PATH + 'train.mat')['train_images']
  train_img = format_img(train_img)
  train_img = np.array(map(lambda a: a.flatten(),train_img),dtype='double')

  # If there was one saving grace to these homeworks, it's the fact
  # that at least no one fucked up the training labels
  train_lab = np.ravel(sio.loadmat(DIGIT_PATH + 'train.mat')['train_labels'])

  return train_img,train_lab,test_img

def load_data_spam():
  '''Loads the spam dataset according to the reasonable format we are given'''
  spams = sio.loadmat(SPAM_PATH + 'spam_data.mat')
  train_img = np.array(spams['training_data'],dtype='double')
  train_lab = np.array(np.ravel(spams['training_labels']),dtype='double')
  test_img = np.array(spams['test_data'],dtype='double')
  return train_img,train_lab,test_img

def preprocess_normalize(imgs):
  '''mean = 0 var = 1'''
  def normalize(v):
    mean = np.average(v)
    std = np.std(v)
    return (v-mean)/std
  return np.array(map(normalize,imgs.T)).T

def preprocess_log(imgs):
  '''Xij <- log(Xij + .1)'''
  tr = lambda x: np.log(x+.1)
  return tr(imgs)

def preprocess_bin(imgs):
  '''Xij <- {0,1}, Xij <= Xij'''
  tr = lambda x: np.ceil(x/(x+.01))
  return tr(imgs)

def get_samples(n,img,lab):
  '''Gives you n samples from the digits data '''
  li = range(len(lab))
  random.shuffle(li)
  return (np.array([img[i] for i in li[:n]]),np.array([lab[i] for i in li[:n]]))

def make_df(samples):
  '''Turns the samples into a pandas DataFrame for easier use
  uses the input from get_samples '''
  samples_df = pd.DataFrame(samples[0])
  samples_df['labels'] = samples[1]
  return samples_df

def classify(x,classifiers,classes):
  '''Classifies x using the gaussian model '''
  pr = map(lambda i: classifiers[i](x), classes)
  return classes[np.argmax(pr)]

def classify_multi(xs,classifiers,classes):
  '''Uses the classifier and returns the predicted labels'''
  '''Assumes that xs is nxd, and labels is nx1 '''
  return np.array(map(lambda x: classify(x,classifiers,classes),xs))

def digits(est=False):
  '''Does digits automatically '''
  NUM_TRAIN = training_set_size
  NUM_VAL = 10000
  train_img,train_lab,test_img = load_data_digits()
  

  classifiers = None
  predicted = None

  if test:
    test_img = normalize(test_img)
    test_predicted = classify_multi(test_img,classifiers,list(set(train_lab)))

    test_out = pd.DataFrame(test_predicted)
    test_out.columns = ['category']
    test_out.index = test_out.index + 1
    test_out.index.names = ['id']
    test_out.to_csv('out/digits.csv')

def train_plot_batch(X,y,pre=None,name='3a_batch'):
  NUM_ITER = 100
  global w
  w = np.array([0.0]*X.shape[1],dtype='double')
  if pre is not None:
    X = pre(X)
  risk = np.array([0] * NUM_ITER,dtype='double')
  for i in xrange(NUM_ITER):
    risk[i] = R(w,X,y)
    w = update(w,X,y,EP)
    print 'iteration',i,'risk',risk[i]
  plt.figure()
  plt.plot(risk)
  plt.savefig('out/' + name + '.png')
  return w,risk

def train_plot_stoch(X,y,pre=None,name='3a_stoch',vtr=False):
  NUM_ITER = 100 * X.shape[0]
  global w
  w = np.array([0.0]*X.shape[1],dtype='double')
  if pre is not None:
    X = pre(X)
  risk = np.array([0] * (NUM_ITER/X.shape[0]),dtype='double')
  it = range(X.shape[0])
  random.shuffle(it)
  ep = EP
  for i in xrange(NUM_ITER):
    if i % X.shape[0] == 0:
      risk[i/X.shape[0]] = R(w,X,y)
      print 'iteration',i,'risk',risk[i/X.shape[0]]
    ep = 10**4*EP/(i+1.) if vtr else ep
    w = update_stoc(w,X,y,ep,it[i%X.shape[0]])
  plt.figure()
  plt_x = xrange(NUM_ITER,X.shape[0])
  plt.plot(risk)
  plt.savefig('out/' + name + '.png')
  return w,risk

def quad_kernel(A,B,rho=1):
  '''Quadratic kernel. A and B are matrices. Calculates (AB^T + 1)^2'''
  return (np.dot(A,B.T) + rho) ** 2

def lin_kernel(A,B,rho=1):
  '''Linear kernel. A and B are matrices. Calculates AB^T + 1'''
  return np.dot(A,B.T) + rho

def kernel_update(a,X,y,ep,idx,kernel=quad_kernel,lam=10**-5,rho=1):
  z = np.dot(kernel(X[idx],X,rho), a)
  a[idx] = a[idx] + ep * (y[idx] - s(z))
  return a

def kernel_ridge(a,X,y,ep,idx,kernel=quad_kernel,lam=10**-5,rho=1):
  z = np.dot(kernel(X[idx],X,rho), a)
  a -= lam * a
  a[idx] = a[idx] + ep * (y[idx] - s(z))
  # if ep * (y[idx] - s(z)) < lam:
  #   print 'your learning sucks',  ep * (y[idx] - s(z))
  return a

def train_plot_kernel(X,y,pre=None,name='3a_kernel',kernel=quad_kernel,kernel_update=kernel_ridge,vtr=False,\
    lam=10**-5,rho=1,plot=True):
  NUM_ITER = 20 * X.shape[0]
  global a
  a = np.array([0.0001/X.shape[0]]*X.shape[0],dtype='double')
  if pre is not None:
    X = pre(X)
  risk = np.array([0] * (NUM_ITER/X.shape[0]),dtype='double')
  it = range(X.shape[0])
  random.shuffle(it)
  ep = EP

  for i in xrange(NUM_ITER):
    if i % X.shape[0] == 0 and plot:
      random.shuffle(it)
      w = np.dot(X.T,a)
      risk[i/X.shape[0]] = R(w,X,y)
      print 'iteration',i,'risk',risk[i/X.shape[0]]
    ep = 10**4*EP/(i+1.) if vtr else ep
    a = kernel_update(a,X,y,ep,it[i%X.shape[0]],kernel=kernel,lam=lam,rho=rho)

  if plot:
    plt.figure()
    plt_x = xrange(NUM_ITER,X.shape[0])
    plt.plot(risk)
    plt.savefig('out/' + name + '.png')

  w = np.dot(X.T,a)
  return a,R(w,X,y)

def xvalidate(X,y,pre=preprocess_log,kernel=quad_kernel,kernel_update=kernel_ridge,lam=10**-5,rho=1):
  global a,Xv,Xt,yv,yt
  if pre is not None:
    X = pre(X)
  it = range(X.shape[0])
  random.shuffle(it)
  cutoff = 2*X.shape[0]/3
  Xv = np.array([X[i] for i in it[cutoff:]],dtype='double')
  yv = np.array([y[i] for i in it[cutoff:]],dtype='double')
  Xt = np.array([X[i] for i in it[:cutoff]],dtype='double')
  yt = np.array([y[i] for i in it[:cutoff]],dtype='double')
  ep = EP

  a = np.array([0.0001/Xt.shape[0]]*Xt.shape[0],dtype='double')
  NUM_ITER = 40 * Xt.shape[0]
  for i in xrange(NUM_ITER):
    if i%(X.shape[0]*4) == 0:
      print 'iteration',i
    a = kernel_update(a,Xt,yt,ep,i%Xt.shape[0],kernel=kernel,lam=lam,rho=rho)

  w = np.dot(Xt.T,a)
  return a,R(w,Xv,yv)


def spam():
  NUM_TRAIN = 4172
  NUM_VAL = 1000

  train_img,train_lab,test_img = load_data_spam()

  global X,y,best_rho,a,w
  X = train_img
  y = train_lab

  # train_plot_stoch(X,y,pre=preprocess_normalize,name='3a_norm_stoch')
  # train_plot_stoch(X,y,pre=preprocess_log,name='3a_log_stoch')
  # train_plot_stoch(X,y,pre=preprocess_bin,name='3a_bin_stoch')

  # train_plot_batch(X,y,pre=preprocess_normalize,name='3a_norm_batch')
  # train_plot_batch(X,y,pre=preprocess_log,name='3a_log_batch')
  # train_plot_batch(X,y,pre=preprocess_bin,name='3a_bin_batch')
  # 
  # train_plot_stoch(X,y,pre=preprocess_normalize,name='3a_norm_vtr',vtr=True)
  # train_plot_stoch(X,y,pre=preprocess_log,name='3a_log_vtr',vtr=True)
  # train_plot_stoch(X,y,pre=preprocess_bin,name='3a_bin_vtr',vtr=True)

  best_rho = 1
  bestR = None

  rhos = np.logspace(-10,10,10)
  for rho in rhos:
    print 'rho rho, fight the power',rho
    a,R = train_plot_kernel(X,y,rho=rho,pre=preprocess_log)
    if bestR is None or R < bestR:
      bestR = R
      best_rho = rho
  
  aq,rq = train_plot_kernel(X,y,rho=best_rho,pre=preprocess_log,name='3a_log_quadk')
  al,rl = train_plot_kernel(X,y,rho=best_rho,kernel=lin_kernel,pre=preprocess_log,name='3a_log_link')

  a = aq if rl > rq else al
  w = np.dot(X.T,a)
  
  test_predicted = np.uint(np.round(s(np.dot(test_img,w))))
  spam_out = pd.DataFrame(test_predicted)
  spam_out.columns = ['category']
  spam_out.index = spam_out.index + 1
  spam_out.index.names = ['id']
  spam_out.to_csv('out/spam.csv')










# main

def main():
  spam()

if __name__=='__main__':
  main()

# ti,tl,tti = load_data_spam()
# train_plot_batch(ti,tl,pre=preprocess_log)
# train_plot_batch(ti,tl,pre=preprocess_bin)
