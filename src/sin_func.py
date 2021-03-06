import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time

random.seed(10)

N = 30
noise_rate = 0.3
x_noise_rate = 0.1
epoch = 1000
batch_size = 8#'Full'

noise = random.randn(N, 1)*noise_rate
x_axis_org = np.linspace(-np.pi,np.pi,N).reshape(N, 1) 
base = np.sin(x_axis_org).reshape(N, 1)
x_axis = x_axis_org + random.randn(N,1)*x_noise_rate
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
        self.input = rm.Dense(1)
        self.hidden = rm.Dense(10)
        self.output = rm.Dense(1)

    def forward(self, x):
        return self.output(rm.tanh(self.hidden(self.input(x))))

func_model = mymodel()
optimizer = rm.Adam()#Sgd(0.2, momentum=0.6)
plt.clf()
epoch_splits = 10
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(8, epoch_splits*4))
if batch_size == 'Full':
    batch_size = len(train_x)

curve = [[], []]
neighbor_period = []
for e in range(epoch):
    s = time()
    perm = np.random.permutation(len(train_x))
    batch_loss = []
    for i in range(0, len(train_x), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = train_x[idx]
        batch_y = train_y[idx]
        with func_model.train():
            loss = rm.mean_squared_error(func_model(batch_x), batch_y)
        grad = loss.grad()
        grad.update(optimizer)
        batch_loss.append(loss.as_ndarray()) 
    neighbor_period.append(time()-s)
    curve[0].append(loss.as_ndarray())
    loss = rm.mean_squared_error(func_model(test_x), test_y)
    curve[1].append(loss.as_ndarray())
    if e % epoch_period == epoch_period - 1 or e == epoch:
        current_period =  np.array(neighbor_period).mean()
        neighbor_period = []
        ax_ = ax[e//epoch_period]
        curve_na = np.array(curve)
        ax_[0].text(0,0.5, '{:.2f}sec @ epoch'.format(current_period))
        ax_[0].plot(curve_na[0])
        ax_[0].plot(curve_na[1])
        ax_[0].set_ylim(0,1)
        ax_[0].set_xlim(-2, epoch+10)
        pred_train = func_model(train_x)
        pred_test = func_model(test_x)
        ax_[1].plot(x_axis_org, base, 'k-')
        ax_[1].scatter(x_axis, y_axis, marker='+')
        ax_[1].scatter(train_x, pred_train, c='g', alpha=0.3)
        ax_[1].scatter(test_x, pred_test, c='r', alpha=0.6)
        rmse = np.power(base-func_model(x_axis_org),2).mean()**0.5
        ax_[1].text(0,-1,'RMSEtrue {:.2f}'.format(rmse))
        plt.pause(0.5)
fig.savefig('result/func{}.png'.format(epoch))
plt.pause(3)
