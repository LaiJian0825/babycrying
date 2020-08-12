import numpy as np
import os



def get_labels(path):
    labels = os.listdir(path)
    print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices

def get_train_test(path):
    labels, indices= get_labels(path)


    X = np.load("./npy/"+labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    for i, label in enumerate(labels[1:]):
        x = np.load("./npy/"+label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    return X, y