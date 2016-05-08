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
  u = u.T[:n].T
  s = np.diag(s[:n])
  v = v[:n]
  return np.dot(np.dot(u,s),v)

def MSE(data,red):
  dudes,things = data.shape
  mse = 0
  for i in xrange(dudes):
    for j in xrange(things):
      if data[i,j] == data[i,j]:
        mse += (red[i,j] - data[i,j]) ** 2
  return mse

def validate_pca(red):
  with open(PATH + 'validation.txt','r') as f:
    right = 0.
    total = 0.
    for line in f:
      split = re.split(',',line)
      me_irl = int(split[0]) - 1
      joke_idx = int(split[1]) - 1
      vpref = 2 * int(split[2]) - 1
      pref = np.sign(red[me_irl,joke_idx])
      right += float(pref == vpref)
      total += 1.
  return right/total

def test_pca(data,raw,n):
  red = reduce_dim(data,n)
  mse = MSE(raw,red)
  score = validate_pca(red)
  print mse
  print score

def pca_jokes():
  get_joke_data()
  for i in [2,5,10,20]:
    test_pca(jokes_fmt,jokes,i)

def good_joke_grad(u,v,l,data,raw):
  diff = (np.dot(u,v) - data) * (raw==raw)
  du = 2*np.dot(diff,v.T) + 2*l*u
  dv = 2*np.dot(u.T,diff) + 2*l*v
  return du,dv

# def good_joke_grad(u,v,l,data,raw):
#   du = u-u
#   dv = v-v
#   for i in xrange(data.shape[0]):
#     for j in xrange(data.shape[1]):
#       du[i] += v[j] * (np.dot(u[i],v[j]) - data[i,j]) * (raw[i,j] == raw[i,j])
#       dv[j] += u[i] * (np.dot(u[i],v[j]) - data[i,j]) * (raw[i,j] == raw[i,j])
#   for i in xrange(data.shape[0]):
#     du[i] += 2*u[i]
#   for j in xrange(data.shape[0]):
#     dv[j] += 2*v[j]
#   return du,dv

def init_good_jokes(data,k):
  num_dudes,num_jokes = data.shape
  u = np.array(map(lambda i: (random.random() - .5),xrange(num_dudes * k))).reshape((num_dudes,k))
  v = np.array(map(lambda i: (random.random() - .5),xrange(num_jokes * k))).reshape((k,num_jokes))
  return u,v

EP = 2**-14

def train_jokes(data,k,lam,raw):
  u,v = init_good_jokes(data,k)
  du,dv = good_joke_grad(u,v,lam,data,raw)
  for it in xrange(500):
    if it % 499 == 0:
      print 'LOSS:', np.sum((raw==raw) * (np.dot(u,v) - data)**2) + lam * (np.linalg.norm(u) + np.linalg.norm(v)) 
    du,_ = good_joke_grad(u,v,lam,data,raw)
    if np.sum(du==du) == 0:
      print 'du is nan'
      return u,v
    u = u - EP * du
    _,dv = good_joke_grad(u,v,lam,data,raw)
    if np.sum(dv==dv) == 0:
      print 'dv is nan'
      return u,v
    v = v - EP * dv
  return u,v

def good_joke_pref(u,v):
  return np.dot(u,v)

def validate_good_joke(uv_pref):
  with open(PATH + 'validation.txt','r') as f:
    right = 0.
    total = 0.
    for line in f:
      split = re.split(',',line)
      me_irl = int(split[0]) - 1
      joke_idx = int(split[1]) - 1
      vpref = 2 * int(split[2]) - 1
      pref = np.sign(uv_pref[me_irl,joke_idx])
      right += float(pref == vpref)
      total += 1.
  return right/total

def kaggle(uv_pref,i=''):
  arr = []
  with open(PATH + 'query.txt','r') as f:
    for line in f:
      split = re.split(',',line)
      idx = split[0]
      me_irl = int(split[1]) - 1
      joke_idx = int(split[2]) - 1
      pref = int((np.sign(uv_pref[me_irl,joke_idx]) + 1)/2)
      arr.append(pref)
  global results
  results = pd.DataFrame(arr)
  results.columns = ['category']
  results.index.name = 'id'
  results.index = results.index + 1
  results.to_csv('out/results' + str(i) + '.csv')

def main():
  get_joke_data()
  lam = .1
  global pref_dict,u,v
  pref_dict = dict()
  for lam in np.logspace(-5,-1,20):
    for k in xrange(25):
    # for k in [2]:
      u,v = init_good_jokes(jokes_fmt,k)
      u,v = train_jokes(jokes_fmt,k,lam,jokes)
      pref = good_joke_pref(u,v)
      good = validate_good_joke(pref)
      pref_dict[str(k) + '||' + str(lam)] = (good,pref)

  best = 0.0
  for k in pref_dict.keys():
    if pref_dict[k][0] > best:
      pref = pref_dict[k][1]
  kaggle(pref)

if __name__=='__main__':
  pass
  # main()
