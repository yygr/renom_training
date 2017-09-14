import os
import sys
import pickle

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

import renom as rm
from renom.optimizer import Sgd, Adam
from renom.cuda.cuda import set_cuda_active
from os import path, makedirs, sep

np.random.seed(100)

set_cuda_active(True)

data_dir = "./cifar-10-batches-py/"
if not path.exists(data_dir):
    makedirs(data_dir)
paths = ["data_batch_1", "data_batch_2", "data_batch_3",
         "data_batch_4", "data_batch_5"]

def unpickle(f):
    fo = open(f, 'rb')
    if sys.version_info.major == 2:
        # Python 2.7
        d = pickle.load(fo)
    elif sys.version_info.major == 3:
        # Python 3.4
        d = pickle.load(fo, encoding="latin-1")
    fo.close()
    return d

# Load train data.
data = list(map(unpickle, [os.path.join(data_dir, p) for p in paths]))
train_x = np.vstack([d["data"] for d in data])
train_y = np.vstack([d["labels"] for d in data])

# Load test data.
data = unpickle(os.path.join(data_dir, "test_batch"))
test_x = np.array(data["data"])
test_y = np.array(data["labels"])

# Rehsape and rescale image.
train_x = train_x.reshape(-1, 3, 32, 32)
train_y = train_y.reshape(-1, 1)
test_x = test_x.reshape(-1, 3, 32, 32)
test_y = test_y.reshape(-1, 1)

train_x = train_x / 255.
test_x = test_x / 255.

# Binalize
labels_train = LabelBinarizer().fit_transform(train_y)
labels_test = LabelBinarizer().fit_transform(test_y)

# Change types.
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)
labels_train = labels_train.astype(np.float32)
labels_test = labels_test.astype(np.float32)

N = len(train_x)

class Cifar10(rm.Model):

    def __init__(self):
        super(Cifar10, self).__init__()
        self._l1 = rm.Conv2d(channel=32)
        self._l2 = rm.Conv2d(channel=32)
        self._l3 = rm.Conv2d(channel=64)
        self._l4 = rm.Conv2d(channel=64)
        self._l5 = rm.Dense(512)
        self._l6 = rm.Dense(10)
        self._sd = rm.SpatialDropout(dropout_ratio=0.25)
        self._pool = rm.MaxPool2d(filter=2, stride=2)

    def forward(self, x):
        t1 = rm.relu(self._l1(x))
        t2 = self._sd(self._pool(rm.relu(self._l2(t1))))
        t3 = rm.relu(self._l3(t2))
        t4 = self._sd(self._pool(rm.relu(self._l4(t3))))
        t5 = rm.flatten(t4)
        t6 = rm.dropout(rm.relu(self._l5(t5)))
        t7 = self._l6(t5)
        return t7

# Choose neural network.
network = Cifar10()
#network = sequential
optimizer = Adam()


# Hyper parameters
batch = 128
epoch = 20

learning_curve = []
test_learning_curve = []

for i in range(epoch):
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        train_batch = train_x[perm[j * batch:(j + 1) * batch]]
        responce_batch = labels_train[perm[j * batch:(j + 1) * batch]]

        # Loss function
        network.set_models(inference=False)
        with network.train():
            l = rm.softmax_cross_entropy(network(train_batch), responce_batch)

        # Back propagation
        grad = l.grad()

        # Update
        grad.update(optimizer)
        loss += l.as_ndarray()

    train_loss = loss / (N // batch)

    # Validation
    test_loss = 0
    M = len(test_x)
    network.set_models(inference=True)
    for j in range(M//batch):
        test_batch = test_x[j * batch:(j + 1) * batch]
        test_label_batch = labels_test[j * batch:(j + 1) * batch]
        prediction = network(test_batch)
        test_loss += rm.softmax_cross_entropy(prediction, test_label_batch).as_ndarray()
    test_loss /= (j+1)

    test_learning_curve.append(test_loss)
    learning_curve.append(train_loss)
    print("epoch %03d train_loss:%f test_loss:%f"%(i, train_loss, test_loss))

