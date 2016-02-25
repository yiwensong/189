from matplotlib import pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import linalg as la

STEP = 0.1

def eigen_decomp(X):
  '''X is a matrix we want to eigendecompose'''
  w,v = la.eigh(X)
  w = np.diag(w)
  return w,v

def square_root(X):
  '''Assume X is a SSPD that we want the square root of.
X = RLR*, returns UL^.5U*'''
  L,R = eigen_decomp(X)
  Lsqrt = np.sqrt(L)
  return np.dot(np.dot(R,Lsqrt),np.transpose(R))

def gauss(x,y,mu,sigma):
  tensor = la.inv(sigma)
  A = square_root(tensor)
  if x-1 <= STEP and y-1 <= STEP and x-1 > -STEP and y-1 > -STEP:
    print A
  v = np.dot(A,np.array([x-mu[0],y-mu[1]]))
  return mlab.bivariate_normal(v[0],v[1])

def gauss_diff(x,y,mu1,sigma1,mu2,sigma2):
  return gauss(x,y,mu1,sigma1) - gauss(x,y,mu2,sigma2)

def plot_isocontour(fcn,xmin=-4,xmax=4,ymin=-4,ymax=4,name='example'):
  '''fcn is a function that maps R^2 -> R. Plots the 2D isocontour of fcn'''
  X = np.arange(xmin,xmax,step=STEP,dtype=float)
  Y = np.arange(ymin,ymax,step=STEP,dtype=float)
  Xzip = np.array([X for i in xrange(len(Y))])
  Yzip = np.array(map(lambda a: [a] * len(X),Y))
  Zf = lambda x,y: fcn(x,y)
  Zff = lambda x,y: map(Zf,x,y)
  Z = map(Zff,Xzip,Yzip)
  Z = np.array(Z)
  Z.reshape(len(X),len(X))
  plt.figure()
  plt.grid(True)
  contour_plot = plt.contour(X,Y,Z)
  plt.savefig( 'out/' + name + '.png')

def gauss_wrapper(mu,sigma):
  return lambda x,y: gauss(x,y,mu,sigma)

def gauss_diff_wrapper(mu1,sigma1,mu2,sigma2):
  return lambda x,y: gauss_diff(x,y,mu1,sigma1,mu2,sigma2)

def qa():
  mu = [1,1]
  sigma = [[2,0],[0,1]]
  fcn = gauss_wrapper(mu,sigma)
  plot_isocontour(fcn,name='2A')

def qb():
  mu = [-1,2]
  sigma = [[3,1],[1,2]]
  fcn = gauss_wrapper(mu,sigma)
  plot_isocontour(fcn,name='2B')

def qc():
  mu1 = [0,2]
  mu2 = [2,0]
  sigma = [[1,1],[1,2]]
  fcn = gauss_diff_wrapper(mu1,sigma,mu2,sigma)
  plot_isocontour(fcn,name='2C')

def qd():
  mu1 = [0,2]
  mu2 = [2,0]
  sigma1 = [[1,1],[1,2]]
  sigma2 = [[3,1],[1,2]]
  fcn = gauss_diff_wrapper(mu1,sigma1,mu2,sigma2)
  plot_isocontour(fcn,name='2D')

def qe():
  mu1 = [1,1]
  mu2 = [-1,-1]
  sigma1 = [[1,0],[0,2]]
  sigma2 = [[2,1],[1,2]]
  fcn = gauss_diff_wrapper(mu1,sigma1,mu2,sigma2)
  plot_isocontour(fcn,name='2E')

 
def main():
  qa()
  qb()
  qc()
  qd()
  qe()

if __name__=='__main__':
  main()
