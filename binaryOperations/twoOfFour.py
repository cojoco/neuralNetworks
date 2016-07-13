#!/usr/bin/env python3

"""This is copied verbatim from
    http://iamtrask.github.io/2015/07/12/basic-python-network/
    and altered for python3
    """

import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,0,0,1], [0,0,1,0,1],
              [0,1,0,0,1], [0,1,0,1,1], [0,1,1,0,1], [0,1,1,1,1],
              [1,0,0,0,1], [1,0,0,1,1], [1,0,1,0,1], [1,0,1,1,1],
              [1,1,0,0,1], [1,1,0,1,1], [1,1,1,0,1], [1,1,1,1,1] ])
                
y = np.array([[0,0,0,1,1,0,0,1,1,0,1,0,0,0]]).T

unknown_X = np.array([[0,0,0,1,1], [0,0,1,1,1]])
unknown_y = np.array([[0,1]]).T

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((5,16)) - 1
syn1 = 2*np.random.random((16,16)) - 1
syn2 = 2*np.random.random((16,1)) - 1

for j in range(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))

    # l3 error
    # how much did we miss the target value?
    l3_error = y - l3

    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l3_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l3_delta = l3_error*nonlin(l3,deriv=True)

    # how much did each l2 value contribute to the l3 error (according to the weights)?
    l2_error = l3_delta.dot(syn2.T)
        
    # in what direction is the target l2?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# Test it on some that we held out
unk_l0 = X
unk_l1 = nonlin(np.dot(unk_l0,syn0))
unk_l2 = nonlin(np.dot(unk_l1,syn1))
unk_l3 = nonlin(np.dot(unk_l2,syn2))

print("Output -- Answer")
for out, ans in zip(unk_l3, unknown_y):
    print(out[0], '--', ans[0])
