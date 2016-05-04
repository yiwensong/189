import numpy as np
from scipy import io as sio
import pandas as pd

import random
import re

PATH = 'data/joke_data/'

def get_joke_data():
  global jokes,jokes_fmt
  jokes = sio.loadmat(PATH + 'joke_train.mat')['train']
  jokes_fmt = np.nan_to_num(jokes)

def n_nearest_neighbors(n,neighbors,me):
  dist = np.linalg.norm(neighbors - me, axis=1)
  return neighbors[dist.argsort()[:n+1]]

nn_cache = dict()
def nnn_query(neighbors,me_irl,joke_idx,n):
  if neighbors[me_irl][joke_idx] != 0:
    return np.sign(neighbors[me_irl][joke_idx])
  try:
    nnn = nn_cache[me_irl + neighbors.shape[0] * n]
  except:
    nnn = n_nearest_neighbors(n,neighbors,neighbors[me_irl])
    nn_cache[me_irl + neighbors.shape[0] * n] = nnn
  pref = np.sign(np.sum(nnn,axis=0))[joke_idx]
  if pref == 0:
    return np.sign(random.random() - .5)
  return pref

def validate(neighbors,n):
  with open(PATH + 'validation.txt','r') as f:
    right = 0.
    total = 0.
    for line in f:
      split = re.split(',',line)
      me_irl = int(split[0]) - 1
      joke_idx = int(split[1]) - 1
      vpref = 2 * int(split[2]) - 1
      pref = nnn_query(neighbors,me_irl,joke_idx,n)
      right += float(pref == vpref)
      total += 1.
  return right/total

def nnn_jokes():
  get_joke_data()
  global d
  d = dict()
  for i in [10,100,1000]:
    d[i] = validate(jokes_fmt,i)

def reduce_dim(data,n):
  u,s,v = np.linalg.svd(data,full_matrices=False)
  b = v[:n]
  return np.dot(np.dot(data,b.T),b)
