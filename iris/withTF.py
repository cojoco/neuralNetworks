#! /usr/bin/env python3

"""This will solve the iris classification problem using TensorFlow"""

import pickle
import numpy as np
import tensorflow as tf


def to_categorical(matrix, num_cats):
    """Converts the matrix to a one-hot matrix with num_cats categories"""
    cats = []
    new_matrix = np.zeros((matrix.shape[0], num_cats))
    for row in matrix:
        for item in row:
            if item not in cats:
                cats.append(item)
    for i, row in enumerate(matrix):
        new_row = np.zeros(num_cats)
        index = cats.index(row[0])
        new_row[index] = 1
        new_matrix[i] = new_row
    return new_matrix


data = pickle.load(open('irisData.pickle', 'rb'))

#Input
inputData = np.array([[d['sepLength'],d['sepWidth'],d['petLength'],d['petWidth']] for d in data])

#Expected output
outputData = np.array([[d['type'].strip()] for d in data])
outputData = to_categorical(outputData, 3)

#BEGIN TENSORFLOW STUFF

sess = tf.InteractiveSession()

x_1 = tf.placeholder(tf.float32, shape=[None, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

W_1 = tf.Variable(tf.zeros([4,3]))
W_2 = tf.Variable(tf.zeros([3,3]))
b_1 = tf.Variable(tf.zeros([3]))
b_2 = tf.Variable(tf.zeros([3]))

sess.run(tf.initialize_all_variables())

x_2 = tf.sigmoid(tf.matmul(x_1,W_1) + b_1)

y = tf.nn.softmax(tf.matmul(x_2,W_2) + b_2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(10000):
    train_step.run(feed_dict={x_1: inputData, y_: outputData})
    if i % 1000 == 0:
        print(accuracy.eval(feed_dict={x_1: inputData, y_: outputData}))

prediction = tf.argmax(y,1)

#Print accuracy and the actual predictions
print(accuracy.eval(feed_dict={x_1: inputData, y_: outputData}))
print(prediction.eval(feed_dict={x_1: inputData}))
