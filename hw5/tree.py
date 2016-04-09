#!/user/bin/python2

import numpy as np
import pandas as pd

from scipy import io as sio

import random
from multiprocessing import Pool

NUM_PROCS = 2
MAX_LEVELS = 1000
FOREST_DENSITY = 100
POINTS_PER_TREE_RATIO = 5000

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
  # print 'best_split size of input',dataframe.shape[0]
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

def random_columns(dataframe):
  entries = dataframe.columns
  entries = filter(lambda n: n != 'label',entries)
  new_cols = np.array(entries)
  random.shuffle(new_cols)
  new_cols = new_cols[:int(np.ceil(np.sqrt(len(new_cols))))]
  new_cols = np.concatenate((new_cols,['label']))
  return new_cols

def random_points(dataframe,n):
  return map(lambda i: random.randint(0,dataframe.shape[0]-1),xrange(n))

def forest_tree(dataframe):
  xs = random_points(dataframe,dataframe.shape[0])
  ys = random_columns(dataframe)
  dataframe = dataframe.iloc[xs][ys]
  tree = train_tree(dataframe)
  print 'making a tree in forest!',ys
  return tree

def train_validate_check_forest(data):
  train = data[0]
  validate = data[1]
  test = data[2]
  tree = forest_tree(train)
  label_v = lambda i: get_label_tree(tree,validate.iloc[i])
  label_t = lambda i: get_label_tree(tree,test.iloc[i])
  validate_results = map(label_v,xrange(validate.shape[0]))
  test_results = map(label_t,xrange(test.shape[0]))
  return [validate_results,test_results]

def forest(train,validate,test,pool):
  pool = Pool(processes=NUM_PROCS)
  output = pool.map(train_validate_check_forest,[(train,validate,test)] * FOREST_DENSITY)
  return np.array(output)

# def generate_and_train(dataframe,col_bag=True,data_bag=True):
#   if col_bag:
#     new_cols = random_columns(dataframe)
#   if data_bag:
#     dataframe = dataframe.iloc[random_points(dataframe,dataframe.shape[0])]
#   print 'training tree with columns:',reduce(lambda a,b: str(a) + ', ' + str(b),new_cols)
#   return (new_cols,train_tree(dataframe[new_cols]))
# 
# def train_random_forest(dataframe):
#   pool = Pool(processes=NUM_PROCS)
#   forest = pool.map(generate_and_train,[dataframe] * FOREST_DENSITY)
#   return forest
# 
# def get_label_tree_pass(arg):
#   tree = arg[0]
#   data = arg[1]
#   return get_label_tree(tree,data)
# 
# def get_label_partial_tree(tree,dataset,cols):
#   print 'getting labels for tree with columns:',reduce(lambda a,b: str(a) + ', ' + str(b),cols)
#   cols = filter(lambda i: True if i != 'label' else False,cols)
#   args = map(lambda i: (tree,dataset[cols].iloc[i]),xrange(dataset.shape[0]))
#   pool = Pool(processes=NUM_PROCS)
#   parallel_return = pool.map(get_label_tree_pass,args)
#   return np.array(parallel_return)
# 
# def get_label_forest(forest,data):
#   labels = np.array(map(lambda i: get_label_partial_tree(forest[i][1],data,forest[i][0]),xrange(len(forest))))
#   labels = np.average(labels,axis=0)
#   return labels

def census():
  train = pd.DataFrame.from_csv('census_data/train_data.csv')
  train = train[train.columns[train.columns != 'fnlwgt']]
  train,validate = get_train_validate(train)
  tree = train_tree(train)
  
  get_lab = lambda i: get_label_tree(tree,validate.iloc[i])
  results = map(get_lab,xrange(validate.shape[0]))
  
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  
  test = pd.DataFrame.from_csv('census_data/test_data.csv')
  get_lab_test = lambda i: get_label_tree(tree,test.iloc[i])
  test_results = map(get_lab_test,xrange(test.shape[0]))
  results_df = make_results_csv(test_results,'census')

def spam():
  spams = sio.loadmat('spam-dataset/spam_data.mat')
  train = spams['training_data']
  label = spams['training_labels']
  test = spams['test_data']
  train = pd.DataFrame(train)
  train['label'] = label.T
  
  train,validate = get_train_validate(train)
  tree = train_tree(train)
  
  get_lab = lambda i: get_label_tree(tree,validate.iloc[i])
  results = map(get_lab,xrange(validate.shape[0]))
  
  score = np.sum(results == validate['label'])/float(validate.shape[0])
  print 'validation correct rate:',score
  
  test = pd.DataFrame(test)
  get_lab_test = lambda i: get_label_tree(tree,test.iloc[i])
  test_results = map(get_lab_test,xrange(test.shape[0]))
  results_df = make_results_csv(test_results,'spam')

def census_forest():
  global train,validate,test
  global results,test_results
  train = pd.DataFrame.from_csv('census_data/train_data.csv')
  train = train[train.columns[train.columns != 'fnlwgt']]
  train,validate = get_train_validate(train)
  test = pd.DataFrame.from_csv('census_data/test_data.csv')
  # global forest
  # forest = train_random_forest(train)
  # 
  # results = get_label_forest(forest,validate)
  # 
  # score = np.sum(results == validate['label'])/float(validate.shape[0])
  # print 'validation correct rate:',score
  # 
  # test_results = get_label_forest(forest,test)
  # results_df = make_results_csv(test_results,'census')

def spam_forest():
  spams = sio.loadmat('spam-dataset/spam_data.mat')
  train = spams['training_data']
  label = spams['training_labels']
  test = spams['test_data']
  train = pd.DataFrame(train)
  train['label'] = label.T
  
  train,validate = get_train_validate(train)
  # forest = train_random_forest(train)
  # 
  # results = get_label_forest(forest,validate)
  # 
  # score = np.sum(results == validate['label'])/float(validate.shape[0])
  # print 'validation correct rate:',score
  # 
  # test = pd.DataFrame(test)
  # test_results = get_label_forest(forest,test)
  # results_df = make_results_csv(test_results,'spam')

def main():
  pass
  # census_forest()
  # spam_forest()

if __name__=='__main__':
  main()
