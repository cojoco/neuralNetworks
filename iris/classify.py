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
    e_x = np.exp(x)
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    return signal


def sigDeriv(x):
    """Gets the derivative, given that we're using the sigmoid to squash"""
    return x * (1 - x)


def softDeriv(x):
    """Gets the derivative, given that we're using the softmax to squash"""
    e_x = np.exp(x)
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    Jacob = - signal[:, :, None] * signal[:, None, :]
    iy, ix = np.diag_indices_from(Jacob[0])
    Jacob[:, iy, ix] = signal * (1. - signal)
    return Jacob.sum(axis=1)


def err(x, y):
    """Returns 1 if the values are not close to the same, 0 otherwise"""
    if x <= y+.25 and x >= y-.25:
        return 0
    return 1


def listErr(x, y):
    """Returns 1 if the biggest number in x does not match the 1 in y"""
    if x.index(max(x)) != y.index(1):
        return 1
    return 0


def npArrErr(x, y):
    """Returns a numpy array with 1 in the row for error, 0 otherwise"""
    argmaxArr = np.argmax(x, axis=1) - np.argmax(y, axis=1)
    newArr = np.zeros((150, 1))
    for i in range(argmaxArr.shape[0]):
        if argmaxArr[i] == 0:
            newArr[i,0] = 0
        else:
            newArr[i,0] = 1
    return newArr


def rmse(x, y):
    """Returns the root mean squared error of the two lists"""
    return np.sqrt(((x - y) ** 2).mean())


# Input, including bias
X = np.array([[float(d['sepLength']), float(d['sepWidth']),
               float(d['petLength']), float(d['petWidth']), 1] for d in data])

y = np.zeros((len(data), 3))
for i, d in enumerate(data):
    if d['type'].strip() == 'Iris-setosa':
        y[i] = [1, 0, 0]
    elif d['type'].strip() == 'Iris-versicolor':
        y[i] = [0, 1, 0]
    elif d['type'].strip() == 'Iris-virginica':
        y[i] = [0, 0, 1]
    else:
        raise Exception('Invalid iris type')

# Get some random weights from -1 to 1
syn0 = 2 * np.random.random((5,16)) - 1
syn1 = 2 * np.random.random((16,16)) - 1
syn2 = 2 * np.random.random((16,3)) - 1

for _ in range(5000):

    # Forward feeding
    l0 = X

    l1 = sigmoid(np.dot(l0, syn0))

    l2 = sigmoid(np.dot(l1, syn1))

    l3 = softmax(np.dot(l2, syn2))

    # Backward propagation
    l3_error = npArrErr(l3, y)

    if (_ % 1000) == 0:
        print("error is", str(np.mean(np.abs(l3_error))))

    l3_delta = l3_error * softDeriv(l3)

    l2_error = l3_delta.dot(syn2.T)

    l2_delta = l2_error * sigDeriv(l2)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigDeriv(l1)
    
    # Update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)

print("Output")
print(l3)
