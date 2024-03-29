import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import pandas as pd

from nn_help import *

DIGIT_PATH = 'dataset/'
TRAIN_SAMPLES = 50000

ITERATIONS = 20

def format_img(img):
  img = np.swapaxes(img,0,2)
  img = np.swapaxes(img,1,2)
  img = img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
  return img

def nn_label_format(i):
  '''index i should be .85 and .15 elsewhere'''
  ret = [.15] * NUM_OUTPUTS
  ret[i] = .85
  return ret

def load_data_digits():
  ''' Loads the digit dataset according to the format
  that we are given'''
  # In this one, the test_img are in format of (10000,784)
  test_img = sio.loadmat(DIGIT_PATH + 'test.mat')['test_images']

  train_img = sio.loadmat(DIGIT_PATH + 'train.mat')['train_images']
  train_img = format_img(train_img)
  train_img = np.array(map(lambda a: a.flatten(),train_img),dtype='double')
  test_img  = np.array(map(lambda a: a.flatten(), test_img),dtype='double')

  test_img  = np.ceil(test_img  / (test_img  + .0001))
  train_img = np.ceil(train_img / (train_img + .0001))

  # row_sums = np.linalg.norm(train_img,axis=1) + .0001
  # train_img = (train_img.T/row_sums[:np.newaxis]).T
  # row_sums = np.linalg.norm(test_img ,axis=1) + .0001
  # test_img  = (test_img.T /row_sums[:np.newaxis]).T

  train_lab = np.ravel(sio.loadmat(DIGIT_PATH + 'train.mat')['train_labels'])
  train_lab = np.array(map(nn_label_format,train_lab))

  return train_img,train_lab,test_img

def get_samples(n,img,lab):
  '''Gives you n samples from the digits data '''
  li = range(len(lab))
  random.shuffle(li)
  return (np.array([img[i] for i in li[:n]]),np.array([lab[i] for i in li[:n]]))

def get_label(features,hidden,output):
  _,__,___,z = nn_output(features,hidden,output)
  return np.argmax(z)

def main():
  global hidden,output,img,lab,test,vimg,vlab
  img,lab,test = load_data_digits()
  hidden,output = initialize()

  available = img.shape[0]
  img,lab = get_samples(img.shape[0],img,lab)
  
  vimg = img[TRAIN_SAMPLES:]
  vlab = lab[TRAIN_SAMPLES:]
  img = img[:TRAIN_SAMPLES]
  lab = lab[:TRAIN_SAMPLES]

  rate = .002
  idx = 0
  iteration = 0
  val_rate = 0.0
  
  train_errors = [0.0] * (ITERATIONS * TRAIN_SAMPLES / 1000)
  validation_errors = [0.0] * (ITERATIONS * TRAIN_SAMPLES / 1000)

  while iteration < ITERATIONS * TRAIN_SAMPLES:
    if idx % TRAIN_SAMPLES == 0:
      img,lab = get_samples(TRAIN_SAMPLES,img,lab)

    if iteration % 1000 == 0:
      print iteration

      lbl = [''] * (available - TRAIN_SAMPLES)
      for i in xrange(available - TRAIN_SAMPLES):
        lbl[i] = get_label(vimg[i],hidden,output)
      lbl = np.array(lbl)
      vlabel_int = np.array(map(np.argmax,vlab))
      val_rate = np.sum(lbl == vlabel_int)/float(lbl.shape[0])

      train_lbl = [''] * (TRAIN_SAMPLES)
      for i in xrange(TRAIN_SAMPLES):
        train_lbl[i] = get_label(img[i],hidden,output)
      train_lbl = np.array(train_lbl)
      label_int = np.array(map(np.argmax,lab))
      tval_rate = np.sum(train_lbl == label_int)/float(train_lbl.shape[0])

      train_errors[ iteration / 1000 ] = 1.0 - tval_rate
      validation_errors[ iteration / 1000 ] = 1.0 - val_rate

      print "Test rate:",tval_rate
      print "Validation rate:",val_rate
      print "Test Error:",iteration/1000,train_errors[ iteration / 1000 ]
      print "Validation Error:",iteration/1000,validation_errors[ iteration / 1000 ]

    hidden,output = backprop(img[idx%TRAIN_SAMPLES],lab[idx%TRAIN_SAMPLES],hidden,output,learning_rate=rate,loss=cross_ent_loss_grad)
    idx = (idx + 1) % TRAIN_SAMPLES

    if (idx % (TRAIN_SAMPLES/4)) == 0:
      rate = rate * .95

    iteration += 1

  tr_err_line, = plt.plot(train_errors)
  v_err_line,  = plt.plot(validation_errors)
  plt.legend(handles=[tr_err_line,v_err_line],labels=['Training Error','Validation Error'])
  plt.savefig('out/errors.svg')

  lbl = [''] * (available - TRAIN_SAMPLES)
  for idx in xrange(available - TRAIN_SAMPLES):
    lbl[idx] = get_label(vimg[idx],hidden,output)
  lbl = np.array(lbl)
  vlabel_int = np.array(map(np.argmax,vlab))
  print "Validation rate:",np.sum(lbl == vlabel_int)/float(lbl.shape[0])

  do_test()

  print 'done'


def do_test():
  tlbl = [''] * len(test)
  for idx in xrange(len(test)):
    tlbl[idx] = get_label(test[idx],hidden,output)
  tlbl = np.array(tlbl)

  global results
  results = pd.DataFrame(tlbl)
  results.columns = np.array(['category'])
  results.index += 1
  results.index.name = 'id'
  results.to_csv('out/nn.csv')
  return results


if __name__=='__main__':
  main()
