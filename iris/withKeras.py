#!/usr/bin/env python3

"""This is an attempt to solve the iris classification problem using Keras"""

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

data = pickle.load(open('irisData.pickle', 'rb'))

X = np.array([[d['sepLength'],d['sepWidth'],d['petLength'],d['petWidth']] for d in data])

y = np.array([[d['type'].strip()] for d in data])

model = Sequential([
    Dense(16, input_dim=4),
    Activation('sigmoid'),
    Dense(16),
    Activation('softmax')])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y = to_categorical(y, 3)

model.fit(X, y, nb_epoch=10, batch_size=5)

score = model.evaluate(X, y, batch_size=5)

print(score)
