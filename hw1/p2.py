# HW1 Part 2

from scipy import io as sio
import numpy as np
from sklearn import svm
from sklearn import metrics
import random
from matplotlib import pyplot as plt
import pandas as pd

NUM_CLASSES = 10
LEARN_SIZES = [100,200,500,1000,2000,5000,10000]
VERIF_SIZE = 10000
DATA_LOC = 'data/spam-dataset/'

zip_star = lambda mat: np.array(zip(*mat))
format_img = lambda img: zip_star(map(zip_star,img))

# Load the data
spams = sio.loadmat(DATA_LOC + 'spam_data.mat')
train_img = spams['training_data']
train_lab = np.ravel(spams['training_labels'])
test_img = spams['test_data']

# Format the data into something that's actually usable. Jesus.
# Good news! The data is actually useable this time.

# Find how many data points there are
DATA_SIZE = train_lab.shape[0]

# Generate a verification set
def k_folds(k,size=DATA_SIZE):
  '''Generates k folds, gives 2d array of DATA_SIZE/k indices'''
  selection = range(DATA_SIZE)
  random.shuffle(selection)
  fold_size = size/k
  folds = [''] * k
  for i in range(k):
    folds[i] = selection[i*fold_size:(i+1)*fold_size]
  return folds

def learn():
  NUM_FOLDS = 10
  XV_SIZE = DATA_SIZE/10
  FOLD_SIZE = XV_SIZE/NUM_FOLDS
  C_VALS = map(lambda a:2**a,xrange(-10,11))
  best = -1
  best_C = None
  # Loop through C-values
  for cval in C_VALS:
    print 'cval',cval
    # Generate folds
    folds = k_folds(NUM_FOLDS,XV_SIZE)
    succ = range(NUM_FOLDS)
    # Loop through folds
    for i in xrange(NUM_FOLDS):
      print 'fold',i
      # Make validation
      v_sel = folds[i]
      v_set_img = [train_img[x] for x in v_sel]
      v_set_lab = [train_lab[x] for x in v_sel]
      # Make train
      t_sel = np.concatenate((np.ndarray.flatten(np.array(folds[0:i],dtype='int')),\
          np.ndarray.flatten(np.array(folds[i+1:],dtype='int'))))
      t_set_img = [train_img[x] for x in t_sel]
      t_set_lab = [train_lab[x] for x in t_sel]
      # Make model
      l_fit = svm.SVC(C=cval,kernel='linear')
      l_fit.fit(t_set_img,t_set_lab)
      # Test model
      p_lab = l_fit.predict(v_set_img)
      succ[i] = sum(map(lambda a,b: 1 if a==b else 0,p_lab,v_set_lab))/float(FOLD_SIZE)
      print succ[i]
    # Average model fit
    succ = np.average(succ)
    print 'success rate',succ
    # Take max score/best C
    if succ > best:
      best = succ
      best_C = cval
  # Save the best C value
  f = open('plots/spam_bestC.csv','w')
  f.write(str(best_C))
  f.close()
  # Predict test data
  f_fit = svm.SVC(C=best_C,kernel='linear')
  keg_train = k_folds(1,20000)[0]
  f_fit.fit([train_img[x] for x in keg_train],[train_lab[x] for x in keg_train])
  return f_fit

def fit(f_fit):
  global keg_pred
  keg_pred = f_fit.predict(test_img)
  df = pd.DataFrame(keg_pred)
  df.columns = ['category']
  df.index = df.index + 1
  df.index.names = ['id']
  df.to_csv('plots/spam.csv')

model = learn()
fit(model)
