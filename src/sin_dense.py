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
epoch = 100
batch_size = 5#'Full'

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

class mymodel_recursive(rm.Model):
    def __init__(
        self, input_shape, 
        output_shape, 
        growth_rate=12, 
        depth=3,
        dropout=False):
        self.depth = depth
        self.dropout = dropout
        if depth != 1:
            under_growth_rate = input_shape + growth_rate
            self.under_model = mymodel(
                under_growth_rate,
                output_shape,
                growth_rate = growth_rate,
                depth = depth - 1)
        else:
            self.output = rm.Dense(output_shape)
        self.batch = rm.BatchNormalize()
        self.conv = rm.Dense(growth_rate)

    def forward(self, x):
        hidden = self.batch(x)
        hidden = rm.tanh(hidden)
        hidden = self.conv(hidden)
        if self.dropout:
            hidden = rm.dropout(hidden)
        hidden = rm.concat(x, hidden)
        if self.depth != 1:
            return self.under_model(hidden)
        #print(hidden.shape)
        return self.output(hidden)

class mymodel(rm.Model):
    def __init__(self, input_shape, output_shape,
        growth_rate = 12,
        depth = 3,
        dropout = False,
        ):
        self.growth_rate = growth_rate
        self.depth = depth
        self.dropout = dropout
        self.input = rm.Dense(input_shape)
        self.output = rm.Dense(output_shape)
        parameters = []
        for _ in range(depth):
            parameters.append(rm.BatchNormalize())
            parameters.append(rm.Dense(growth_rate))
        self.hidden = rm.Sequential(parameters)
    
    def forward(self, x):
        layers = self.hidden._layers
        hidden = self.input(x)
        i = 0
        for _ in range(self.depth):
            main_stream = hidden
            hidden = layers[i](main_stream)
            i += 1
            hidden = rm.tanh(hidden)
            hidden = layers[i](hidden)
            i += 1
            if self.dropout:
                hidden = rm.dropout(hidden)
            hidden = rm.concat(main_stream, hidden)
        #print(hidden.shape)
        return self.output(hidden)

# growth_rate = 2
# depth = 10
# mymodel perform 1 epoch @ 1.9 sec
# mymodel_recursive perform 1 epoch @ 1.3 sec
func_model = mymodel(1, 1, growth_rate=2, depth=10, dropout=False)
optimizer = rm.Sgd(lr=0.2, momentum=0.6)
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
            loss = rm.mean_squared_error(func_model(train_x), train_y)
        grad = loss.grad()
        grad.update(optimizer)
        batch_loss.append(loss.as_ndarray()) 
    neighbor_period.append(time()-s)
    curve[0].append(np.array(batch_loss).mean())
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
        ax_[1].plot(x_axis, base, 'k-')
        ax_[1].scatter(x_axis, y_axis, marker='+')
        ax_[1].scatter(train_x, pred_train, c='g', alpha=0.3)
        ax_[1].scatter(test_x, pred_test, c='r', alpha=0.6)
        plt.pause(0.5)
fig.savefig('result/dense.png')
plt.pause(3)
