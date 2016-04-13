#!/user/bin/python2

import numpy as np
import pandas as pd

from scipy import io as sio

import argparse
import random
from multiprocessing import Pool
import os
import time

NUM_PROCS = 6
MAX_LEVELS = 10000
CENSUS_FOREST_DENSITY = 300
SPAM_FOREST_DENSITY = 300
POINTS_PER_TREE_RATIO = 5000

def random_columns(dataframe):
  entries = dataframe.columns
  entries = filter(lambda n: n != 'label',entries)
  new_cols = np.array(entries)
  random.shuffle(new_cols)
  new_cols = new_cols[:int(np.ceil(np.sqrt(len(new_cols))))]
  new_cols = np.concatenate((new_cols,['label']))
  return new_cols

def random_points(dataframe,n):
  df0 = dataframe[dataframe['label']==0]
  df1 = dataframe[dataframe['label']==1]
  num0 = map(lambda i: random.randint(0,df0.shape[0]-1),xrange(n/2))
  num1 = map(lambda i: random.randint(0,df1.shape[0]-1),xrange(n/2))
  sel0 = df0.iloc[num0]
  sel1 = df1.iloc[num1]
  return sel0.append(sel1)

def data_cleanup(dataframe,ref):
  for c in dataframe.columns:
    if dataframe[c].dtype == 'O':
      mode = ref[c].mode()[0]
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

def best_split(dataframe,columns=None):
  if columns is None:
    columns = dataframe.columns
  # print 'best_split size of input',dataframe.shape[0]
  labels = dataframe['label']
  not_labels = np.array(filter(lambda l: l != 'label',columns))

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

  def traverse_numeric(self,data,verbose=False):
    global d
    d = data
    if data[self.feature] >= self.value:
      # go left
      ret = self.left.traverse(data,verbose)
    else:
      # go right
      ret = self.right.traverse(data,verbose)
    return ret

  def traverse_obj(self,data,verbose=False):
    if data[self.feature] == self.value:
      # go left
      ret = self.left.traverse(data,verbose)
    else:
      # go right
      ret = self.right.traverse(data,verbose)
    return ret

  def traverse(self,data,verbose=False):
    if verbose:
      try:
        print 'going to:',self.feature,self.value,data[self.feature]
      except:
        print 'leafed! it\'s a',self.value
    if self.feature_type is None:
      return self.value
    elif self.feature_type != 'O':
      ret = self.traverse_numeric(data,verbose)
    else:
      ret = self.traverse_obj(data,verbose)
    return ret


def train_tree(dataframe,level=0,randomize=False):
  if randomize:
    cols = random_columns(dataframe)
  else:
    cols = None
  if level >= MAX_LEVELS:
    if dataframe.shape[0] == 1:
      leaf = tree_node(None,None,dataframe.label.iloc[0],None,None)
    try:
      majority = dataframe['label'].mode()[0]
    except:
      return tree_node(None,None,random.randint(0,1),None,None)
    leaf = tree_node(None,None,majority,None,None)
    return leaf
  feature,value,_,pos,neg = best_split(dataframe,cols)
  if pos.shape[0] == 0 or neg.shape[0] == 0:
    try:
      majority = dataframe['label'].mode()[0]
    except:
      return tree_node(None,None,random.randint(0,1),None,None)
    leaf = tree_node(None,None,majority,None,None)
    return leaf

  # print 'level',level,'node splitting',feature,'at',value
  if np.average(pos['label']) == 0:
    left = tree_node(None,None,0,None,None)
  elif np.average(pos['label']) == 1:
    left = tree_node(None,None,1,None,None)
  else:
    left = train_tree(pos,level+1,randomize)

  if np.average(neg['label']) == 0:
    right = tree_node(None,None,0,None,None)
  elif np.average(neg['label']) == 1:
    right = tree_node(None,None,1,None,None)
  else:
    right = train_tree(neg,level+1,randomize)

  ftype = dataframe[feature].dtype
  root = tree_node(feature,ftype,value,left,right)
  return root

def get_label_tree(tree,data):
  return tree.traverse(data)

def get_train_validate(dataframe):
  n = dataframe.shape[0]
  indices = range(n)
  random.shuffle(indices)
  train = dataframe.iloc[indices[:2*n/3]]
  validate = dataframe.iloc[indices[2*n/3:]]
  return train,validate

def make_results_csv(test_results,thing):
  results_df = pd.DataFrame.from_dict(test_results)
  results_df.columns = np.array(['category'])
  results_df.index.name = 'id'
  results_df.index = results_df.index + 1
  results_df.to_csv('out/' + thing + '.csv')
  return results_df

def forest_tree(dataframe):
  new_df = random_points(dataframe,dataframe.shape[0])
  # ys = random_columns(dataframe)
  print 'making a tree in forest!'
  # new_df = new_df[ys]
  tree = train_tree(new_df,randomize=True)
  return tree

def train_validate_check_forest(data):
  start = time.clock()

  train = data[0]
  validate = data[1]
  test = data[2]
  tree = forest_tree(train)

  label_v = lambda i: get_label_tree(tree,validate.iloc[i])
  label_t = lambda i: get_label_tree(tree,test.iloc[i])
  validate_results = map(label_v,xrange(validate.shape[0]))
  test_results = map(label_t,xrange(test.shape[0]))

  print 'time elpased:',time.clock() - start
  return np.array(validate_results),np.array(test_results),(tree.feature,tree.value)

def forest(train,validate,test,FOREST_DENSITY):
  global v,t,output
  # pool = Pool(processes=NUM_PROCS)
  output = map(train_validate_check_forest,[(train,validate,test)] * FOREST_DENSITY)
  v = [''] * FOREST_DENSITY
  t = [''] * FOREST_DENSITY
  featval = [''] * FOREST_DENSITY
  scores = [0] * FOREST_DENSITY
  for i in xrange(FOREST_DENSITY):
    v[i] = output[i][0]
    t[i] = output[i][1]
    featval[i] = output[i][2]
    scores[i] = 1
  total_score = sum(scores)
  for i in xrange(FOREST_DENSITY):
    t[i] = np.array(t[i],dtype='float') * scores[i]/total_score
    v[i] = np.array(v[i],dtype='float') * scores[i]/total_score
  v_res = np.array(np.round(np.sum(v,axis=0)),dtype='int')
  t_res = np.array(np.round(np.sum(t,axis=0)),dtype='int')
  return v_res,t_res,featval

def census(target='census'):
  global censuses
  train = pd.DataFrame.from_csv('census_data/train_data.csv')
  # data_cleanup(train,train)
  censuses = train
  train = train[train.columns[train.columns != 'fnlwgt']]
  train,validate = get_train_validate(train)
  global tree
  tree = train_tree(train)
  
  get_lab = lambda i: get_label_tree(tree,validate.iloc[i])
  results = map(get_lab,xrange(validate.shape[0]))
  
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  
  test = pd.DataFrame.from_csv('census_data/test_data.csv')
  # data_cleanup(test,train)
  get_lab_test = lambda i: get_label_tree(tree,test.iloc[i])
  test_results = map(get_lab_test,xrange(test.shape[0]))
  results_df = make_results_csv(test_results,target)
  return results_df

def spam(target='spam'):
  global train
  spams = sio.loadmat('spam-dataset/spam_data.mat')
  train = spams['training_data']
  label = spams['training_labels']
  test = spams['test_data']
  train = pd.DataFrame(train)
  train['label'] = label.T
  
  train,validate = get_train_validate(train)
  global tree
  tree = train_tree(train)
  
  get_lab = lambda i: get_label_tree(tree,validate.iloc[i])
  results = map(get_lab,xrange(validate.shape[0]))
  
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  
  test = pd.DataFrame(test)
  get_lab_test = lambda i: get_label_tree(tree,test.iloc[i])
  test_results = map(get_lab_test,xrange(test.shape[0]))
  results_df = make_results_csv(test_results,target)
  return results_df

def census_forest(target='census'):
  global train,validate,test
  global results,test_results
  train = pd.DataFrame.from_csv('census_data/train_data.csv')
  # data_cleanup(train,train)
  train = train[train.columns[train.columns != 'fnlwgt']]
  train,validate = get_train_validate(train)
  test = pd.DataFrame.from_csv('census_data/test_data.csv')
  # data_cleanup(test,train)
  
  results,test_results,top_splits = forest(train,validate,test,CENSUS_FOREST_DENSITY)
  with open('out/census_forest.out','w') as f:
    for split in top_splits:
      f.write(split[0] + ',' + str(split[1]) + '\n')
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  results_df = make_results_csv(test_results,target)
  return results_df

def spam_forest(target='spam'):
  global train,validate,test
  global results,test_results
  spams = sio.loadmat('spam-dataset/spam_data.mat')
  train = spams['training_data']
  label = spams['training_labels']
  test = spams['test_data']
  test = pd.DataFrame(test)
  train = pd.DataFrame(train)
  train['label'] = label.T
  train,validate = get_train_validate(train)

  train.columns = np.array(train.columns,dtype='|S21')
  validate.columns = np.array(validate.columns,dtype='|S21')
  test.columns = np.array(test.columns,dtype='|S21')

  results,test_results,top_splits = forest(train,validate,test,SPAM_FOREST_DENSITY)
  with open('out/spam_forest.out','w') as f:
    for split in top_splits:
      f.write(split[0] + ',' + str(split[1]) + '\n')
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  results_df = make_results_csv(test_results,target)
  return results_df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c','--censusforest',help='do census forest',action='store_true')
  parser.add_argument('-s','--spamforest',help='do spam forest',action='store_true')
  parser.add_argument('-C','--censustree',help='do census tree',action='store_true')
  parser.add_argument('-S','--spamtree',help='do spam tree',action='store_true')
  parser.add_argument('-t','--target',help='output file location')

  args = parser.parse_args()
  
  global census_results,census_tree_results,spam_results,spam_tree_results

  if args.target is not None:
    if args.censusforest:
      census_results = census_forest(args.target)

    if args.spamforest:
      spam_results = spam_forest(args.target)

    if args.censustree:
      census_tree_results = census(args.target)

    if args.spamtree:
      spam_tree_results = spam(args.target)
  else:
    if args.censusforest:
      census_results = census_forest()

    if args.spamforest:
      spam_results = spam_forest()

    if args.censustree:
      census_tree_results = census()

    if args.spamtree:
      spam_tree_results = spam()

if __name__=='__main__':
  main()
