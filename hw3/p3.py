import random
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

# we wrote shit to transform stuff already in part 2
import p2

average = None
Sigma = None
eigenvalues = None
eigenvectors = None

def generate_samples():
  global samples
  x1 = map(random.normalvariate,[3]*100,[3]*100)
  x2 = map(random.normalvariate,[4]*100,[2]*100)
  x2 = map(lambda a,b: .5*a + b, x1,x2)
  samples = (x1,x2)
  return (x1,x2)

def qa(samples):
  global average
  average = map(np.average,samples)

def qb(samples):
  global Sigma
  Sigma = np.cov(samples)

def qc(matrix = Sigma):
  global eigenvalues,eigenvectors
  eigenvalues,eigenvectors = la.eigh(matrix)
 
def qd(samples,evectors=eigenvectors):
  plt.figure()
  plt.plot(samples[0],samples[1],'o')
  plt.grid(True)
  xcenter = average[0]
  ycenter = average[1]
  print xcenter,ycenter
  print evectors
  plt.arrow(xcenter,ycenter,evectors[0][0]*eigenvalues[0],evectors[0][1]*eigenvalues[0],head_width=.15,head_length=.3)
  plt.arrow(xcenter,ycenter,evectors[1][0]*eigenvalues[1],evectors[1][1]*eigenvalues[1],head_width=.15,head_length=.3)
  plt.savefig('out/3D.png')

def qe(samples,evals=eigenvalues,evectors=eigenvectors):
  global A
  invsigma = la.inv(Sigma)
  A = p2.square_root(invsigma)
  samples = np.array(samples)
  samples -= np.transpose(np.array([average]*100))
  samples_normal = np.dot(A,samples)
  plt.figure()
  plt.plot(samples_normal[0],samples_normal[1],'o')
  plt.grid(True)
  plt.savefig('out/3E.png')

def main():
  generate_samples()
  qa(samples)
  qb(samples)
  qc(Sigma)
  qd(samples,eigenvectors)
  qe(samples,eigenvalues,eigenvectors)

if __name__=='__main__':
  main()
