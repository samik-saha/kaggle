import json
import numpy as np

def load_dataset():
    file = open('data\\train.json', 'r')
    datastore = json.load(file)
    x_train = np.empty((1605, 5625))
    i=0
    for item in datastore:
        band_2 = np.array(item['band_2'])
        band_1 = np.array(item['band_1'])
        x_train[i,:] = band_2
        i+=1

    x_train = np.reshape(x_train, (1605,75,75))

    y_train = np.array((5625))
    for item in datastore:
        y_train = np.vstack([y_train, item['is_iceberg']])

    return x_train, y_train

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y