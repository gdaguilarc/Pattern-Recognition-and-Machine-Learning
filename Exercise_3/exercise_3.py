from visual_classification import *
from evaluation_prediction import *
from cifar_10_read_data import *
from tqdm import tqdm
import numpy as np

test_labels = DATA['test_labels']
"""
# Training Phase
parameters_learned = cifar_10_bayes_learn()

print("Naive assumption normal distribution")
# For Naive assumption normal distribution
predicted_labels = []
for vector in tqdm(DATA['test_data']):
    predicted_labels.append(
        cifar_10_bayes_classify(vector, parameters_learned))


# Testing with data
print("PREDICTED DATA:", np.array(predicted_labels).view())

accuracy = evaluation(predicted_labels, test_labels)
print("Accuracy Naive Assumption Normal Distribution: {0}".format(accuracy))
"""

# ---------------------------------------------------------------------------

parameters_learned = cifar_10_bayes_learn("covariance")

print("Multivariate Distribution")
# Multivariate Normal Distribution
predicted_labels_mvn = []
for vector in tqdm(DATA['test_data']):
    predicted_labels_mvn.append(
        cifar_10_bayes_classify_mvn(vector, parameters_learned))


# Testing with data
print("PREDICTED DATA:", np.array(predicted_labels_mvn))

accuracy_mvn = evaluation(predicted_labels_mvn, test_labels)
print("Accuracy Multivariate Distribution: {0}".format(accuracy_mvn))
