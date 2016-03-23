import numpy as np

X = np.array([[0,3,1],[1,3,1],[0,1,1],[1,1,1]],dtype='double')
y = np.array([1,1,0,0],dtype='double')

def s(x):
  sx = 1./(1.+np.exp(-x))
  sx -= ((sx-.5) * 10**-10)
  return sx

def r(w,x,y):
 return y*np.log(s(np.dot(w.T,x))) + (1-y)*np.log(1.-s(np.dot(w.T,x)))

def R(w,x,y):
  r = lambda x,y: y*np.log(s(np.dot(w.T,x))) + (1-y)*np.log(1.-s(np.dot(w.T,x)))
  return -sum(map(r,x,y))

def gradr(w,x,y):
  return (y - s(np.dot(x.T,w)))*x

def gradR(w,x,y):
  gradr = lambda x,y: (y - s(np.dot(x.T,w)))*x
  return sum(map(gradr,x,y))

iters = 3

w = np.array([[-2,1,0]] * iters,dtype='double')

ep = 1

def update(w,x,y,ep):
  return w + ep * gradR(w,x,y)

def update_stoc(w,x,y,ep,i):
  return w + ep * gradr(w,x[i],y[i])

def main():
  for i in xrange(iters-1):
    w[i+1] = update(w[i],X,y,ep)
  
  for i in xrange(iters):
    print 'iteration',i
    print 'w',w[i]
    print 'mu',s(np.dot(X,w[i]))
    print 'R',R(w[i],X,y)
    print ''

if __name__=='__main__':
  main()
