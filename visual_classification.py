from cifar_10_read_data import *
from scipy.stats import norm
from scipy.stats import multivariate_normal
from random import randrange
import numpy as np

DATA = load_all()
BATCHES = np.split(DATA['data'], 5)
CLASSES = 10

CHANNELS = 12
IMAGE_SIZE = 256

"""
CHANNELS = 12 FOR 8x8 images size cuts
CHANNELS = 24 FOR 4x4 images size cuts

"""


def cifar_10_features(x, operation='mean', size=IMAGE_SIZE, channels=CHANNELS):
    VECTOR_SIZE = len(x)

    # Handles vectors and matrix
    if(len(np.shape(x)) == 1):
        channels_splited = np.split(x, channels)
    else:
        channels_splited = np.split(x, channels, axis=1)

    f = []
    for arr in channels_splited:
        if operation == 'mean':
            f.append(np.mean(arr))
        else:
            temp = []
            for vector in arr:
                temp.append(np.mean(vector))

            f.append(np.std(temp))

    return f


def cifar_10_bayes_learn(operation='variance', classes=CLASSES, labels=DATA['labels'], F=DATA['data']):

    # sorted
    sorted_arr = {}
    # The returned object
    parameters = {}

    for i in range(0, len(F)):
        vector = F[i]
        label = str(labels[i])

        if label in sorted_arr.keys():
            sorted_arr[label].append(vector)
        else:
            sorted_arr[label] = []
            sorted_arr[label].append(vector)

    for key in sorted_arr.keys():
        sorted_arr[key] = np.array(sorted_arr[key])

    for index in range(0, classes):
        label = str(index)
        parameters[label] = []
        parameters[label].append(cifar_10_features(
            sorted_arr[label]))
        if operation == 'variance':
            parameters[label].append(cifar_10_features(
                sorted_arr[label], operation))
        else:
            means = []
            for vector in sorted_arr[label]:
                means.append(cifar_10_features(vector))
            means = np.array(means)
            parameters[label].append(np.cov(means.T))
    return parameters


def cifar_10_bayes_classify(f, params, channels=CHANNELS, p=(1/CLASSES)):
    result_label = 0
    max_prob = 0
    mean_f = cifar_10_features(f)

    for key in params.keys():
        means = params[str(key)][0]
        variances = params[str(key)][1]

        result = norm.pdf(mean_f[0], means[0], variances[0])*norm.pdf(
            mean_f[1], means[1], variances[1])*norm.pdf(mean_f[2], means[2], variances[2])*p

        if max_prob < result:
            max_prob = result
            result_label = key

    return int(result_label)


def cifar_10_bayes_classify_mvn(f, params, classes=CLASSES, p=1/10):
    result_label = randrange(0, 10)
    max_prob = 0
    mean_f = cifar_10_features(f)
    for key in params.keys():
        result = (multivariate_normal.pdf(
            mean_f, params[str(key)][0], params[str(key)][1]))*p

        if max_prob < result:
            max_prob = result
            result_label = key

    return int(result_label)


def split_array(arr, n):
    if(len(arr) % n == 0):
        i = 0
        result = []
        temp = []
        for element in arr:
            temp.append(element)
            if i == n - 1:
                result.append(temp)
                temp = []

    return result
