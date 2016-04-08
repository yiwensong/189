#!/user/bin/python2

import numpy as np
import pandas as pd

import random

def data_cleanup(dataframe):
  for c in dataframe.columns:
    if dataframe[c].dtype == 'O':
      mode = dataframe[c].mode()
      dataframe[c] = map(lambda s: mode if s == '?' else s,dataframe[c])

def entropy(x):
  if len(x) == 0:
    return 0
  x = np.array(x,dtype='double')
  x = x/np.sum(x)
  if x[0] == 1 or x[1] == 1:
    return 0
  return sum( - x * np.log2(x) )

def find_subframe_entropy(subframe):
  cnts = subframe.groupby('label').count()[subframe.columns[0]]
  return entropy(cnts)

def numeric_split(dataframe,feature,value):
  geq = dataframe[dataframe[feature] >= value]
  let = dataframe[dataframe[feature] < value]
  h = (geq.shape[0] * find_subframe_entropy(geq) + let.shape[0] * find_subframe_entropy(let)) / (geq.shape[0] + let.shape[0])
  if geq.shape==0 or let.shape == 0:
    h = 10000
  return h,geq,let

def nonnumeric_split(dataframe,feature,group):
  geq = dataframe[dataframe[feature] == group]
  let = dataframe[dataframe[feature] != group]
  h = (geq.shape[0] * find_subframe_entropy(geq) + let.shape[0] * find_subframe_entropy(let)) / (geq.shape[0] + let.shape[0])
  if geq.shape==0 or let.shape == 0:
    h = 10000
  return h,geq,let

def best_split(dataframe):
  print 'best_split size of input',dataframe.shape[0]
  labels = dataframe['label']
  not_labels = np.array(filter(lambda l: l != 'label',dataframe.columns))

  # feature name / entropy pair
  all_best_split = None
  for feature in not_labels:
    best_split = None
    if dataframe[feature].dtype == 'O':
      # Split by object
      for group in dataframe[feature].unique():
        h,yeah,nope = nonnumeric_split(dataframe,feature,group)
        if best_split is None or best_split[1] < h:
          best_split = (group,h,yeah,nope)
    else:
      # Split by value
      for value in dataframe[feature].unique():
        h,geq,let = numeric_split(dataframe,feature,value)
        if best_split is None or h < best_split[1]:
          best_split = (value,h,geq,let)
    if all_best_split is None or best_split[1] < all_best_split[2]:
      all_best_split = (feature,best_split[0],best_split[1],best_split[2],best_split[3])
  return all_best_split
    

class tree_node:
  def __init__(self,feature,feature_type,value,left,right):
    self.feature = feature
    self.feature_type = feature_type
    self.value = value
    self.left = left
    self.right = right

  def split(self,dataframe):
    if self.feature_type is None:
      return self.value
    elif self.feature_type != 'O':
      _,left,right = numeric_split(dataframe,self.feature,self.value)
    else:
      _,left,right = nonnumeric_split(dataframe,self.feature,self.value)
    return left,right

  def traverse_numeric(self,data):
    if data[self.feature] >= self.value:
      # go left
      ret = self.left.traverse(data)
    else:
      # go right
      ret = self.right.traverse(data)
    return ret

  def traverse_obj(self,data):
    if data[self.feature] == self.value:
      # go left
      ret = self.left.traverse(data)
    else:
      # go right
      ret = self.right.traverse(data)
    return ret

  def traverse(self,data):
    if self.feature_type is None:
      return self.value
    elif self.feature_type != 'O':
      ret = self.traverse_numeric(data)
    else:
      ret = self.traverse_obj(data)
    return ret

MAX_LEVELS = 4

def train_tree(dataframe,level=0):
  if level >= MAX_LEVELS:
    if dataframe.shape[0] == 1:
      leaf = tree_node(None,None,dataframe.label.iloc[0],None,None)
    try:
      majority = dataframe['label'].mode()[0]
    except:
      return tree_node(None,None,random.randint(0,1),None,None)
    leaf = tree_node(None,None,majority,None,None)
    return leaf
  feature,value,_,pos,neg = best_split(dataframe)
  if pos.shape[0] == 0 or neg.shape[0] == 0:
    majority = dataframe['label'].mode()[0]
    leaf = tree_node(None,None,majority,None,None)
    return leaf

  print 'level',level,'node splitting',feature,'at',value
  if np.average(pos['label']) == 0:
    left = tree_node(None,None,0,None,None)
  elif np.average(pos['label']) == 1:
    left = tree_node(None,None,1,None,None)
  else:
    left = train_tree(pos,level+1)

  if np.average(neg['label']) == 0:
    right = tree_node(None,None,0,None,None)
  elif np.average(neg['label']) == 1:
    right = tree_node(None,None,1,None,None)
  else:
    right = train_tree(neg,level+1)

  ftype = dataframe[feature].dtype
  root = tree_node(feature,ftype,value,left,right)
  return root

def get_label_tree(tree,data):
  return tree.traverse(data)

train = pd.DataFrame.from_csv('census_data/train_data.csv')
train = train[train.columns[train.columns != 'fnlwgt']]
tree = train_tree(train)
