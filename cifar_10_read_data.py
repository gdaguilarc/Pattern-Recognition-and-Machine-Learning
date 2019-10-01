DATA_DIR = './cifar-10-batches-py'
BATCHES = 5


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def batch_data(number=1, directory=DATA_DIR):
    loaded = unpickle("{0}/data_batch_{1}".format(directory, number))
    data = {}
    data['data'] = loaded[b'data']
    data['labels'] = loaded[b'labels']
    return data


def test_data(directory=DATA_DIR):
    loaded = unpickle("{0}/test_batch".format(directory))
    data = {}
    data['data'] = loaded[b'data']
    data['labels'] = loaded[b'labels']
    return data


def load_all(directory=DATA_DIR, number_batches=BATCHES):
    import numpy as np
    data = {}
    for index in range(0, number_batches):
        batch = batch_data(index + 1, directory)
        batch_labels = batch['labels']
        batch_data_vectors = batch['data']
        data['data'] = np.concatenate((data['data'], batch_data_vectors), axis=0) if 'data' in data.keys(
        ) else batch_data_vectors
        data['labels'] = np.concatenate((data['labels'], batch_labels), axis=0) if 'labels' in data.keys(
        ) else batch_labels

    test = test_data(directory)
    data['test_data'] = test['data']
    data['test_labels'] = test['labels']
    return data
