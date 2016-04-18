#!/usr/bin/python2

import numpy as np

s = lambda x: 1./(1. + np.exp(-x))
ds = lambda x: s(x) * (1-s(x))

tanh = lambda x: (1. - np.exp(-2. * x))/(1. + np.exp(-2. * x))
dtanh = lambda x: 1 - tanh(x) ** 2





## TEST CODE ##
for i in np.arange(-10,10,.1):
  if (tanh(i) - np.tanh(i)) > np.abs(np.tanh(i))*.01:
    print 'ERROR IN TANH: max error bound',np.abs(np.tanh(i))*.01,'actual error',(tanh(i) - np.tanh(i))
