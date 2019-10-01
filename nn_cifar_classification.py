'''

 Guillermo David Aguilar Castilleja
 Tampere Univerity of Technology
 Exercise 4: Visual classification with neural networks (CIFAR-10)

 '''

import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.layers import *
from keras.models import Sequential
import numpy as np
from cifar_10_read_data import *
import keras
import tensorflow as tf

from keras.optimizers import Adam


config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


# Import Cifar_1o data

DATA = load_all()
training_data = DATA['data']
training_labels = DATA['labels']
test_data = DATA['test_data']
test_labels = DATA['test_labels']

# Translate label data into single binary output


def single_binary(labels):
    binary_arr = []
    for label in labels:
        temp = []
        for i in range(10):
            if(i == label):
                temp.append(1)
            else:
                temp.append(0)
        binary_arr.append(temp)
    return np.array(binary_arr)


training_labels = single_binary(training_labels)
test_labels = single_binary(test_labels)


def flattening_img(data):
    result = []
    for image in data:
        result.append(np.reshape(image, (32, 32, 3)))
    return np.array(result)


training_data = flattening_img(training_data)
test_data = flattening_img(test_data)
print(training_data)


model = Sequential()
model.add(InputLayer(input_shape=[32, 32, 3]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1,
                 padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=5, padding="same"))

model.add(Conv2D(filters=50, kernel_size=5, strides=1,
                 padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=5, padding="same"))

model.add(Conv2D(filters=80, kernel_size=5, strides=1,
                 padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=5, padding="same"))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="sigmoid"))
model.add(Dropout(rate=0.5))
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation="softmax"))

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='mse', metrics=['accuracy'])
model.fit(x=training_data, y=training_labels, epochs=20, batch_size=1000)

results = model.evaluate(test_data, test_labels)
print('test loss, test acc:', results)


y_pred = model.predict(test_data)


def normalize_output(labels):
    labels_result = []
    for label in labels:
        labels_result.append(get_max(label))
    return labels_result


def get_max(array):
    max_index = 0
    max_value = 0
    for index in range(len(array)):
        if(max_value < array[index]):
            max_value = array[index]
            max_index = index
    return max_index


labels_predicted = normalize_output(y_pred)
print(labels_predicted[:10])

print(DATA['test_labels'][:10])


'''
# Load necessary packages for neural networks

# Model sequential
model = Sequential()
# First hidden layer (we also need to tell the input dimension)
model.add(Dense(100, input_dim=3072, activation='sigmoid'))
# First hidden layer (we also need to tell the input dimension)
model.add(Dense(100, activation='sigmoid'))


# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(10, activation='tanh'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

model.fit(training_data, training_labels, epochs=5, verbose=1)
y_pred = model.predict(test_data)
print(y_pred)
print(test_labels)

'''
