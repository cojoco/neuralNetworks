#!/usr/bin/env python3

"""This is a basic XOR neural network I attempted to recreate from memory
    I then looked at a correct neural network and fixed my errors
    """

import numpy as np

def sigmoid(x):
    """This is our squashing function"""
    return 1 / (1 + np.exp(-x))

def deriv(x):
    """This is returns the derivative of the layer"""
    return x * (1 - x)

# This is our input
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# This is the true output
y = np.array([[0,1,1,0]]).T

# These are our randomly initialized weights for the first layer to the second
syn0 = 2*np.random.random((3,3)) - 1
# These are our randomly initialized weights for the second layer to the third
syn1 = 2*np.random.random((3,1)) - 1

for _ in range(10000):

    # Forward Feeding
    # This is our first layer
    l0 = X

    # This is our second layer
    l1 = sigmoid(np.dot(l0,syn0))

    # This is our third layer
    l2 = sigmoid(np.dot(l1,syn1))

    # Backward Propagation
    # First get the error
    l2_error = y - l2

    # Then find the delta
    l2_delta = l2_error * deriv(l2)

    # Then find the next error
    l1_error = np.dot(l2_delta,syn1.T) # Could also be l2_delta.dot(syn1.T)

    # Then find the next delta
    l1_delta = l1_error * deriv(l1)

    # Then update the weights based on what we found
    syn0 += np.dot(l0.T,l1_delta) # Could also be l0.T.dot(l1_delta)
    syn1 += np.dot(l1.T,l2_delta) # Could also be l1.T.dot(l2_delta)

print("Output")
print(l2)
