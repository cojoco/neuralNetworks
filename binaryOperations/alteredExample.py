#!/usr/bin/env python3

""" This is copied verbatim
    from http://iamtrask.github.io/2015/07/12/basic-python-network/
    and then changed for python3 (print statements, xrange to range)
    and altered as I messed with it.
    """

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1],
                [0,0,0] ])
    
# output dataset            
y = np.array([[0,0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(0)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

print(X)
print(syn0)
print(np.dot(X,syn0))

for iter in range(5):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)

