#!/usr/bin/env python3

"""This uses a neural network to classify the iris dataset"""

import pickle
import numpy as np

data = None

with open('irisData.pickle', 'rb') as ifh:
    data = pickle.load(ifh)


def sigmoid(x):
    """Squashes the value to between 0 and 1"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Squashes a list of values to between 0 and 1"""
    newArr = []
    for val in x:
        newArr.append(np.exp(val)/sum([np.exp(z_k) for z_k in x]))
    return newArr


def sigDeriv(x):
    """Gets the derivative, given that we're using the sigmoid to squash"""
    return x * (1 - x)


def softDeriv(x):
    """Gets the derivative, given that we're using the softmax to squash"""
    #TODO:
    pass


def err(x, y):
    """Returns 1 if the lists are not the same, 0 otherwise"""
    errSum = 0
    for val1, val2 in zip(x, y):
        errSum = val1 - val2 
    return errSum


# Input
X = np.array([[float(d['sepLength']), float(d['sepWidth']),
               float(d['petLength']), float(d['petWidth'])] for d in data])

preY = []
for d in data:
    if d['type'].strip() == 'Iris-setosa':
        preY.append([1,0,0])
    elif d['type'].strip() == 'Iris-versicolor':
        preY.append([0,1,0])
    elif d['type'].strip() == 'Iris-virginica':
        preY.append([0,0,1])
    else:
        raise Exception('Invalid iris type')

# Expected output
y = np.array([preY]).T

# Get some random weights from -1 to 1
syn0 = 2 * np.random.random((4,16)) - 1
syn1 = 2 * np.random.random((16,3)) - 1

for _ in range(50000):

    # Forward feeding
    l0 = X

    l1 = sigmoid(np.dot(l0, syn0))

    l2 = softmax(np.dot(l1, syn1))

    # Backward propagation
    l2_error = sum([(l2_k - y_k)**2 for y_k, l2_k in zip(y, l2)])

    if (_ % 10000) == 0:
        print("error is", str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * softDeriv(l2)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigDeriv(l1)

    # Update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

print("Output")
print(l2)
