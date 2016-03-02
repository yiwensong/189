import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy import io as sio
from scipy import stats
import random
import pandas as pd

from multiprocessing import Pool
from multiprocessing import Lock

import p2

DIGIT_PATH = 'data/digit_dataset/'
SPAM_PATH = 'data/spam_dataset/'
NUM_CLASSES = 10

zip_star = lambda mat: np.array(zip(*mat))
format_img = lambda img: zip_star(map(zip_star,img))
format_img = lambda img: np.array(map(np.transpose,np.transpose(img)))
def format_img(img):
  img = np.swapaxes(img,0,2)
  # img = np.swapaxes(img,1,2)
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
  # train_img = np.transpose(train_img)
  # train_img = np.array(map(np.transpose,np.transpose(train_img)))
  train_img = np.array(map(lambda a: a.flatten(),train_img),dtype='double')

  # If there was one saving grace to these homeworks, it's the fact
  # that at least no one fucked up the training labels
  train_lab = np.ravel(sio.loadmat(DIGIT_PATH + 'train.mat')['train_labels'])

  return train_img,train_lab,test_img

def load_data_spam():
  ''' Loads the spam dataset according to the reasonable format we are given'''
  spams = sio.loadmat(SPAM_PATH + 'spam_data.mat')
  train_img = spams['training_data']
  train_lab = np.ravel(spams['training_labels'])
  test_img = spams['test_data']
  return train_img,train_lab,test_img

def normalize(imgs):
  ''' Turns all values into floating point values between 0 and 1
  by dividing all values by MAX(IMG).
  Takes input nxd'''
  def normalize_helper(img):
    ''' Takes input nparray R^d and normalizes for you'''
    if max(img) == 0:
      return img
    return (img/la.norm(img))
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

# We can just use stats.multivariate_normal
# def train_classifier(label,mus,sigmas):
#   ''' Takes in a label, a mean vector (mu), and a covariance matrix (sigma)
#   and returns a tuple of a list and a matrix. The list represents the indices
#   of the class that matter towards looking at this class, and the matrix
#   represents the transformation to isotropic space. '''
#   mu = mus[label]
#   sigma = sigmas[label]
#   matters = [1]*len(mu)
#   for i in xrange(len(mu)):
#     d = np.diagonal(sigma)[i]
#     if d == 0:
#       matters[i] = 0
#   matters = filter(lambda i: matters[i],xrange(len(mu)))
#   inv_filter = lambda array: np.array([array[i] for i in matters])
#   mu = inv_filter(mu)
#   sigma = inv_filter(sigma)
#   sigma = np.array(map(inv_filter,sigma))
#   sigma_inv = la.inv(sigma)
#   A = p2.square_root(A)
#   return (matters,A)
# 
# def gauss(x,matters,A):
#   ''' Takes in a feature vector (x), a matters vector of features that matter
#   and a transformation matrix A to calculate the probability of P(X=x|label) '''

GAMMA = .001
def train_classifier(label,mus,sigmas):
  ''' Takes in a label, the mus and the sigmas.
  Returns the pdf function of the distribution it gives.'''
  try:
    mvn = stats.multivariate_normal(mus[label],\
        sigmas[label]+np.diag([GAMMA]*len(mus[label])))
  except Exception as e:
    print label
    print mus
    print sigmas
    raise e
  return mvn.logpdf

def classify(x,classifiers,classes):
  ''' Classifies x using the gaussian model '''
  pr = map(lambda i: classifiers[i](x), classes)
  return classes[np.argmax(pr)]

def classify_multi(xs,classifiers,classes):
  ''' Uses the classifier and returns the predicted labels'''
  ''' Assumes that xs is nxd, and labels is nx1 '''
  return np.array(map(lambda x: classify(x,classifiers,classes),xs))

def parallel_pdf(x,label,mus,sigmas):
  ''' The parallelable version of train + classify '''
  mvn = stats.multivariate_normal(mus[label],sigmas[label],allow_singular=True)
  return mvn.logpdf(x)

def parallel_classify(args):
  x,labels,mus,sigmas = args
  pr = map(lambda l: parallel_pdf(x,l,mus,sigmas),labels)
  return labels[np.argmax(pr)]

def parallel_multi(xs,labels,mus,sigmas):
  p = Pool(8)
  args = map(lambda a,b,c,d: (a,b,c,d), xs,[labels]*len(xs),[mus]*len(xs),[sigmas]*len(xs))
  return np.array(p.map(parallel_classify,args))

def plot_sigma(sigma,name = ''):
  print 'plotting...'
  # plt.figure()
  # plt.pcolor(sigma)
  # plt.savefig('out/5C' + str(name) + '.png')

def digits(linear=False,plot=False,smallplot=False,training_set_size=50000,f=None,test=False):
  ''' Does digits automatically '''
  NUM_TRAIN = training_set_size
  NUM_VAL = 10000
  train_img,train_lab,test_img = load_data_digits()
  train_img = normalize(train_img)
  
  train_samples = get_samples(NUM_TRAIN+NUM_VAL,train_img,train_lab)
  v_img,v_lab = (train_samples[0][NUM_TRAIN:],train_samples[1][NUM_TRAIN:])
  train_samples = (train_samples[0][:NUM_TRAIN],train_samples[1][:NUM_TRAIN])
  train_df = make_df(train_samples)
  global Sigmas,sigmas,mus
  mus = find_average(train_df)
  sigmas = find_sigmas(train_df)
  Sigmas = sigmas
  if linear:
    global counts
    counts = train_df.groupby('labels').count()[0]
    weighted = sum(map(lambda i: counts[i]*sigmas[i],xrange(len(counts))))
    weighted = weighted/sum(counts)
    sigmas = np.array([weighted] * len(set(train_lab)))

  classifiers = map(lambda l: train_classifier(l,mus,sigmas),list(set(train_lab)))
  # v_img,v_lab = get_samples(NUM_VAL,train_img,train_lab)
  predicted = classify_multi(v_img,classifiers,list(set(train_lab)))
  # predicted = parallel_multi(v_img,list(set(train_lab)),mus,sigmas)
  if f is None:
    print 'error rate:',(np.sum(predicted != v_lab))/float(NUM_VAL)
  else:
    f.write(str(NUM_TRAIN) + ',' + str((np.sum(predicted != v_lab))/float(NUM_VAL)) + '\n')

  if plot:
    map(plot_sigma,sigmas.values()[:2],xrange(2))
  if smallplot:
    plt.pcolor(train_img[30000].reshape(28,28),cmap=plt.cm.Blues)
    plt.savefig('out/ex.png')

  if test:
    test_img = normalize(test_img)
    test_predicted = classify_multi(test_img,classifiers,list(set(train_lab)))
    # test_predicted = parallel_multi(test_img,list(set(train_lab)),mus,sigmas)

    # For interactive
    global test_out
    test_out = pd.DataFrame(test_predicted)
    test_out.columns = ['category']
    test_out.index = test_out.index + 1
    test_out.index.names = ['id']
    test_out.to_csv('out/digits.csv')

  return (np.sum(predicted != v_lab))/float(NUM_VAL)

def spam(linear=True):
  NUM_TRAIN = 4172
  NUM_VAL = 1000

  train_img,train_lab,test_img = load_data_spam()
  
  global train_df
  global sigmas,mus
  train_samples = get_samples(NUM_TRAIN+NUM_VAL,train_img,train_lab)
  v_img,v_lab = (train_samples[0][NUM_TRAIN:],train_samples[1][NUM_TRAIN:])
  train_samples = (train_samples[0][:NUM_TRAIN],train_samples[1][:NUM_TRAIN])
  train_df = make_df(train_samples)
  mus = find_average(train_df)
  sigmas = find_sigmas(train_df)
  if linear:
    global counts
    counts = train_df.groupby('labels').count()[0]
    weighted = sum(map(lambda i: counts[i]*sigmas[i],xrange(len(counts))))
    weighted = weighted/sum(counts)
    sigmas = np.array([weighted] * len(set(train_lab)))

  classifiers = map(lambda l: train_classifier(l,mus,sigmas),list(set(train_lab)))
  predicted = classify_multi(v_img,classifiers,list(set(train_lab)))
  print 'error rate:',(np.sum(predicted != v_lab))/float(NUM_VAL)

  test_predicted = classify_multi(test_img,classifiers,list(set(train_lab)))

  # For interactive
  global spam_out
  spam_out = pd.DataFrame(test_predicted)
  spam_out.columns = ['category']
  spam_out.index = spam_out.index + 1
  spam_out.index.names = ['id']
  spam_out.to_csv('out/spam.csv')










# main

def main():
  print 'what'
  tr_sizes = [100,200,500,1000,2000,5000,10000,30000,50000]
  lda_dict = dict()
  qda_dict = dict()
  with open('out/5d.csv','w') as f:
    for i in tr_sizes:
      print 'lda',i
      lda_dict[i] = digits(linear=True,training_set_size=i,f=f)
      print 'qda',i
      qda_dict[i] = digits(linear=False,training_set_size=i,f=f)

  log_sz = np.log(np.array(tr_sizes))
  plt.figure()
  plt.plot(log_sz, [lda_dict[i] for i in tr_sizes])
  plt.savefig('out/5D1.png')

  plt.figure()
  plt.plot(log_sz, [qda_dict[i] for i in tr_sizes])
  plt.savefig('out/5D2.png')
  # digits()
  # spam()


  # Uncomment when done debugging
  # test_img = normalize(test_img)

if __name__=='__main__':
  main()















# TEST CODE
# train_img,train_lab,test_img = load_data_digits()
# train_img = normalize(train_img)
# 
# train_samples = get_samples(1000,train_img,train_lab)
# train_df = make_df(train_samples)
# mus = find_average(train_df)
# sigmas = find_sigmas(train_df)
# 
# keys = list(set(train_lab))
