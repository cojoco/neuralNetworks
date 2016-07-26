#!/usr/bin/env python3

"""This is an attempt to solve the iris classification problem using Keras"""

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


def to_categorical(matrix, num_cats):
    """Converts the matrix to a one-hot matrix with num_cats categories"""
    cats = []
    new_matrix = np.zeros((matrix.shape[0], num_cats))
    for row in matrix:
        for item in row:
            print(item)
            if item not in cats:
                cats.append(item)
    for i, row in enumerate(matrix):
        new_row = np.zeros(num_cats)
        index = cats.index(row[0])
        new_row[index] = 1
        new_matrix[i] = new_row
    return new_matrix


data = pickle.load(open('irisData.pickle', 'rb'))

X = np.array([[d['sepLength'],d['sepWidth'],d['petLength'],d['petWidth']] for d in data])

y = np.array([[d['type'].strip()] for d in data])

model = Sequential([
    Dense(16, input_dim=4),
    Activation('sigmoid'),
    Dense(3),
    Activation('softmax')])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y = to_categorical(y, 3)

model.fit(X, y, nb_epoch=10000, batch_size=5)

score = model.evaluate(X, y, batch_size=5)

print(score)
