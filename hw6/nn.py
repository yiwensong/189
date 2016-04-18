import numpy as np
import scipy.io as sio

from nn_help import *

DIGIT_PATH = 'dataset/'
TRAIN_SAMPLES = 55000

ITERATIONS = 20000

def format_img(img):
  img = np.swapaxes(img,0,2)
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

  # col_sums = train_img.sum(axis=0) + .0001
  # train_img = train_img/col_sums[:np.newaxis]
  # test_img  = test_img /col_sums[:np.newaxis]

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

  for idx in xrange(ITERATIONS):
    if idx % 10000 == 0:
      print idx
    hidden,output = backprop(img[idx%TRAIN_SAMPLES],lab[idx%TRAIN_SAMPLES],hidden,output,learning_rate=.0005,loss=mean_sq_loss_grad)

  lbl = [''] * (available - TRAIN_SAMPLES)
  for idx in xrange(available - TRAIN_SAMPLES):
    lbl[idx] = get_label(vimg[idx],hidden,output)

  lbl = np.array(lbl)
  vlabel_int = np.array(map(np.argmax,vlab))
  print "Validation rate:",np.sum(lbl == vlabel_int)/float(lbl.shape[0])


if __name__=='__main__':
  main()
