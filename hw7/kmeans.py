import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io as sio

import random

PATH = 'data/mnist_data/'
OUT  = 'out/'

def load_data_digits():
  global img
  img = sio.loadmat(PATH + 'images.mat')['images']
  img = np.swapaxes(img,0,2)
  img = np.swapaxes(img,1,2)
  img = img.reshape((img.shape[0],img.shape[1]*img.shape[2]))/255.

def assign(img,centers):
  dist = lambda center: np.linalg.norm(img - center,axis=1)
  distances = np.array(map(dist,centers))
  label = np.argmin(distances,axis=0)
  return label

def recenter(img,assignments,k):
  get_avg = lambda lab: np.average(img[assignments==lab], axis=0)
  return np.array(map(get_avg,xrange(k)))

def loss(img,centers,assignments):
  dist = lambda i: np.sum(np.linalg.norm(img[assignments == i] - centers[i],axis=1))
  distances = map(dist,range(len(centers)))
  return np.sum(distances)

def kmeans(img,k):
  global assignments,centers
  n = len(img)
  idx = range(n)
  random.shuffle(idx)
  old_ass = np.array([None] * len(img))
  centers = np.array([img[i] for i in idx[:k]])
  assignments = assign(img,centers)

  while np.sum(assignments != old_ass) > 0:
    old_ass = assignments
    assignments = assign(img,centers)
    centers = recenter(img,assignments,k)

  print 'k:',k,'|| loss:', loss(img,centers,assignments)
  return centers

def main():
  load_data_digits()
  km = lambda k: kmeans(img,k)
  global results
  results = dict()
  
  for i in xrange(10):
    results[5],results[10],results[20] = map(km,[5,10,20])

  results[5] = np.reshape(results[5],(results[5].shape[0],np.sqrt(results[5].shape[1]),np.sqrt(results[5].shape[1])))
  results[10] = np.reshape(results[10],(results[10].shape[0],np.sqrt(results[10].shape[1]),np.sqrt(results[10].shape[1])))
  results[20] = np.reshape(results[20],(results[20].shape[0],np.sqrt(results[20].shape[1]),np.sqrt(results[20].shape[1])))
  for i in [5,10,20]:
    for j in xrange(i):
      plt.figure()
      plt.imshow(results[i][j])
      plt.savefig(OUT + 'MNIST' + str(i) + '-' + str(j) + '.png')
      plt.close()


if __name__=='__main__':
  main()
