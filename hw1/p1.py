# HW1 Part 1

from scipy import io as sio
import numpy as np
from sklearn import svm
from sklearn import metrics
import random
from matplotlib import pyplot as plt

NUM_CLASSES = 10
LEARN_SIZES = [100,200,500,1000,2000,5000,10000]
LEARN_SIZES = []
VERIF_SIZE = 10000
DATA_LOC = 'data/digit-dataset/'

zip_star = lambda mat: np.array(zip(*mat))
format_img = lambda img: zip_star(map(zip_star,img))

# Load the data
test_img = sio.loadmat(DATA_LOC + 'test.mat')['test_images']
train_img = sio.loadmat(DATA_LOC + 'train.mat')['train_images']
train_lab = sio.loadmat(DATA_LOC + 'train.mat')['train_labels']

# Format the data into something that's actually usable. Jesus.
train_img = format_img(train_img)
train_lab = np.ravel(train_lab)
test_img = format_img(test_img)
train_img = map(np.ndarray.flatten,train_img)
test_img = map(np.ndarray.flatten,test_img)

DATA_SIZE = len(train_lab)

# Generate a verification set
selection = range(0,DATA_SIZE)
random.shuffle(selection)
verif = selection[0:VERIF_SIZE]
sel = selection[VERIF_SIZE:]

s = dict()
pred = dict()
actl = dict()

# Try some different sizes for training and see how they do
for l_size in LEARN_SIZES:

  print 'learning and verifying for n =',l_size

  # Select some verification data
  random.shuffle(sel)
  train_selection = sel[0:l_size]

  # Train the classifier
  lin_fit = svm.SVC(kernel='linear')
  lin_fit.fit([train_img[i] for i in train_selection],[train_lab[i] for i in train_selection])

  # Try and predict the set
  pred[l_size] = lin_fit.predict([train_img[i] for i in verif])
  actl[l_size] = [train_lab[i] for i in verif]
  s[l_size] = float(sum(map(lambda a,b: 1 if a==b else 0,pred[l_size],actl[l_size])))/float(VERIF_SIZE)
  print l_size,'success rate:',s[l_size]
  print ''

def p1():
  # Problem 1: Plot training data size versus success rate
  plt.plot(LEARN_SIZES,[s[sz] for sz in LEARN_SIZES])
  plt.xscale('log')
  plt.savefig('plots/p1.png',format='png')

# p1()

def p2():
  # Problem 2: Create a confusion matrix
  confusion = dict()
  for l_size in LEARN_SIZES:
  
    # Find the confusion matrix
    confusion[l_size] = metrics.confusion_matrix(actl[l_size],pred[l_size])

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(confusion[l_size],interpolation='nearest',cmap=plt.cm.Purples)
    w,h = confusion[l_size].shape

    for x in xrange(w):
      for y in xrange(h):
        ax.annotate(str(confusion[l_size][x][y]), xy=(y,x),\
            horizontalalignment='center',\
            verticalalignment='center')

    plt.title('Confusion matrix for ' + str(l_size) + ' samples')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(NUM_CLASSES),xrange(NUM_CLASSES))
    plt.yticks(np.arange(NUM_CLASSES),xrange(NUM_CLASSES))
    plt.savefig('plots/p2_conf_' + str(l_size) + '.png', format='png')

# p2()

def k_folds(k,size=DATA_SIZE):
  '''Generates k folds, gives 2d array of DATA_SIZE/k indices'''
  random.shuffle(selection)
  fold_size = size/k
  folds = [''] * k
  for i in range(k):
    folds[i] = selection[i*fold_size:(i+1)*fold_size]
  return folds

def p3():
  NUM_FOLDS = 10
  XV_SIZE = 10000
  FOLD_SIZE = XV_SIZE/NUM_FOLDS
  C_VALS = [.001,.01,.1,1,10,100,1000]
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
  # Predict test data
  f_fit = svm.SVC(C=best_C,kernel='linear')
  f_fit.fit(train_img,train_lab)
  global keg_pred
  keg_pred = f_fit.predict(test_img)

p3()




