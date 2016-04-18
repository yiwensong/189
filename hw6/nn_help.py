import numpy as np
import scipy.io as sio
import random

NUM_FEATURES = 784
NUM_HIDDEN = 200
NUM_OUTPUTS = 10

def sigmoid(x):
  return 1.0/(1.0 + np.exp(x))

def dsigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))

tanh = np.tanh

def dtanh(x):
  return 1 - tanh(x)**2

def mean_sq_loss(pred,actual):
  loss = np.sum((pred-actual)**2.0)/2.0
  return loss

def cross_ent_loss(pred,actual):
  p = (99./100. * pred + .5 * 1./100.)
  return -sum(actual * np.log(p) + (1-actual) * np.log(1-p))

def mean_sq_loss_grad(pred,actual):
  return pred - actual

def cross_ent_loss_grad(pred,actual):
  p = (99./100. * pred + .5 * 1./100.)
  return -actual/p + (1-actual)/(1-p)

def hidden_output(hidden,features):
  f = np.concatenate((features,[1.]))
  return np.dot(hidden,f)

def output_output(output,hiddens):
  h = np.concatenate((hiddens,[1.]))
  return np.dot(output,h)

def nn_output(features,hidden,output):
  h0 = hidden_output(hidden,features)
  h = tanh(h0)
  z0 = output_output(output,h)
  z = sigmoid(z0)
  return h0,h,z0,z

def output_weight_grad(z,loss_g,h):
  col_grad = loss_g * z * (1.-z)
  h = np.concatenate((h,[1.]))
  return np.outer(col_grad,h)

def hidden_out_grad(z,loss_g,W):
  col_grad = loss_g * z * (1.-z)
  return np.dot(col_grad,W)

def hidden_weight_grad(q,f,h_grad):
  qh = (1.-q**2.) * h_grad[:-1]
  f = np.concatenate((f,[1.]))
  return np.outer(qh,f)

def backprop(features,label,hidden,output,learning_rate=.005,loss=mean_sq_loss_grad):
  h0,h,z0,z = nn_output(features,hidden,output)
  Lz_grad = loss(z,label)
  # print 'mean sq loss',mean_sq_loss(z,label)
  # print 'cross ent loss',cross_ent_loss(z,label)
  
  output_update = output_weight_grad(z,Lz_grad,h)

  Lh_grad = hidden_out_grad(z,Lz_grad,output)
  hidden_update = hidden_weight_grad(h,features,Lh_grad)

  output = output + learning_rate * output_update
  hidden = hidden + learning_rate * hidden_update

  return hidden,output

def initialize():
  get_rand = lambda _: random.normalvariate(0,1)
  get_row = lambda s: np.array(map(get_rand,xrange(s)))
  normalize_row = lambda c: c/np.sum(c)

  rows = map(get_row,[NUM_FEATURES + 1] * NUM_HIDDEN)
  hidden = np.array(map(normalize_row,rows))

  rows = map(get_row,[NUM_HIDDEN + 1] * NUM_OUTPUTS)
  output = np.array(map(normalize_row,rows))

  return hidden,output
