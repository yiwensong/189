import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy import io as sio
import random
import pandas as pd

import p2

DIGIT_PATH = 'data/digit_dataset/'

def load_data_digits():
  ''' Loads the digit dataset according to the shitty format
  that we are given'''
  # In this one, the test_img are in format of (10000,784)
  # Fucking inconsistent as shit
  test_img = sio.loadmat(DIGIT_PATH + 'test.mat')['test_images']

  # Unfortunately, the training set is still fucked up
  train_img = sio.loadmat(DIGIT_PATH + 'train.mat')['train_images']
  train_img = np.array(map(np.transpose,np.transpose(train_img)))
  train_img = np.array(map(lambda a: a.flatten(),train_img),dtype='double')

  # If there was one saving grace to these homeworks, it's the fact
  # that at least no one fucked up the training labels
  train_lab = np.ravel(sio.loadmat(DIGIT_PATH + 'train.mat')['train_labels'])

  return train_img,train_lab,test_img

def normalize(imgs):
  ''' Turns all values into floating point values between 0 and 1
  by dividing all values by MAX(IMG).
  Takes input nxd'''
  def normalize_helper(img):
    ''' Takes input nparray R^d and normalizes for you'''
    return img/max(img)
  return np.array(map(normalize_helper,imgs),dtype='double')

def get_samples(n,img,lab):
  ''' Gives you n samples from the digits data '''
  li = range(len(lab))
  random.shuffle(li)
  return (np.array([img[i] for i in li[:n]]),np.array([lab[i] for i in li[:n]]))

def make_df(samples):
  ''' Turns the samples into a pandas DataFrame for easier use
  uses the input from get_samples '''
  samples_df = pd.DataFrame(samples[0])
  samples_df['labels'] = samples[1]
  return samples_df

def find_average(samples_df):
  ''' Takes in a pandas DataFrame of all the features and the labels and
  finds the average for each label.
  The output is a DataFrame where rows are labels and columns are features '''
  samples_gb = samples_df.groupby('labels')
  samples_gb_sums = samples_gb.sum().transpose() # Transposes for ease of use
  samples_gb_counts = np.array(samples_gb.count()[0]) # A row of the counts per label
  ret_df = samples_gb_sums/samples_gb_counts # Divide per label
  return ret_df

def find_sigma(samples_df):
  ''' Takes in a df with only 1 label and returns the covariance '''
  data = samples_df[samples_df.columns[:-1]]
  data = np.array(data).transpose()
  return np.cov(data)

def find_sigmas(samples_df):
  ''' Takes in pandas DataFrame for all features and labels and computes the
  covariance matrix for each label.
  The output is probably going to be a dictionary with the (k,v) being
  (label,Sigma). No guarantees, though. '''
  labels = list(set(samples_df['labels']))
  sigma = map(lambda x: find_sigma(samples_df.query('labels == @x')),labels)
  d = dict()
  map(lambda k,v: d.__setitem__(k,v),labels,sigma)
  return d

def main():
  train_img,train_lab,test_img = load_data_digits()
  train_img = normalize(train_img)
  
  train_samples = get_samples(1000,train_img,train_lab)
  train_df = make_df(train_samples)
  mus = find_average(train_df)
  sigmas = find_sigmas(train_df)

  keys = list(set(train_lab))


  # Uncomment when done debugging
  # test_img = normalize(test_img)

if __name__=='__main__':
  # main()
  pass

train_img,train_lab,test_img = load_data_digits()
train_img = normalize(train_img)

train_samples = get_samples(1000,train_img,train_lab)
train_df = make_df(train_samples)
mus = find_average(train_df)
sigmas = find_sigmas(train_df)

keys = list(set(train_lab))
