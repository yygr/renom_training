import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random

random.seed(10)

N = 30
noise_rate = 0.3
epoch = 100

noise = random.randn(N)*noise_rate
x_axis = np.linspace(-np.pi,np.pi,N)
base = np.sin(x_axis)
y_axis = base+noise
x_axis = x_axis.reshape(N, 1)
y_axis = y_axis.reshape(N, 1)
idx = random.permutation(N)
train_idx = idx[::2]
test_idx = idx[1::2]
train_x = x_axis[train_idx]
train_y = y_axis[train_idx]
test_x = x_axis[test_idx]
test_y = y_axis[test_idx]

class mymodel(rm.Model):
    def __init__(self):
        self.layer_list = [
            rm.Dense(1),
            #rm.BatchNormalize(),
            rm.Dense(10),
            rm.Sigmoid(),
            rm.Dense(1)
        ]
        #self.input = rm.Dense(1)
        #self.hidden = rm.Dense(10)
        #self.output = rm.Dense(1)

    def forward(self, x):
        hidden = self.layer_list[0](x)
        for item in self.layer_list[1:]:
            hidden = item(hidden)
        return hidden

func_model = mymodel()
optimizer = rm.Adam()
plt.clf()
epoch_splits = 10
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(4, epoch_splits))

curve = [[], []]
for e in range(epoch):
    with func_model.train():
        loss = rm.mean_squared_error(func_model(train_x), train_y)
    grad = loss.grad()
    grad.update(optimizer)
    curve[0].append(loss.as_ndarray())
    loss = rm.mean_squared_error(func_model(test_x), test_y)
    curve[1].append(loss.as_ndarray())
    if e % epoch_period == epoch_period - 1 or e == epoch:
        set_trace()
        ax_ = ax[e//epoch_period]
        curve_na = np.array(curve)
        ax_[0].plot(curve_na[0])
        ax_[0].plot(curve_na[1])
        pred_train = func_model(train_x)
        pred_test = func_model(test_x)
        ax_[1].plot(x_axis, base, 'k-')
        ax_[1].scatter(x_axis, y_axis, marker='+')
        ax_[1].scatter(train_x, pred_train, c='g', alpha=0.3)
        ax_[1].scatter(test_x, pred_test, c='r', alpha=0.8)
        plt.pause(0.5)
fig.savefig('result/func_kai.png')
plt.pause(3)
