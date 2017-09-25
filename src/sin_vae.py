import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
np.set_printoptions(precision=3)

random.seed(10)

N = 30
noise_rate = 0.3
x_noise_rate = 0.1
epoch = 1000
batch_size = 8#'Full'
vae_learn = True

noise = random.randn(N)*noise_rate
x_axis_org = np.linspace(-np.pi,np.pi,N) 
base = np.sin(x_axis_org)
x_axis = x_axis_org + random.randn(N)*x_noise_rate
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

class EncoderDecoder(rm.Model):
    def __init__(self,
        input_shape,
        output_shape,
        units = 10,
        depth = 3):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.depth = depth
        parameters = []
        for _ in range(depth):
            parameters.append(rm.Dense(units))
        self.hidden = rm.Sequential(parameters)
        self.input = rm.Dense(input_shape)
        self.multi_output = False
        if isinstance(self.output_shape, tuple):
            self.multi_output = True
            parameters = []
            for _ in range(output_shape[0]):
                parameters.append(rm.Dense(output_shape[1]))
            self.output = rm.Sequential(parameters)
        else:
            self.output = rm.Dense(output_shape)
    
    def forward(self, x):
        layers = self.hidden._layers
        hidden = self.input(x)
        for i in range(self.depth):
            hidden = layers[i](hidden)
            hidden = rm.relu(hidden)
        if self.multi_output:
            layers = self.output._layers
            outputs = []
            for i in range(self.output_shape[0]):
                outputs.append(layers[i](hidden))
            return outputs
        hidden = rm.relu(hidden)
        return self.output(hidden)

class Vae(rm.Model):
    def __init__(self, enc, dec, output_shape = 1):
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        self.z = enc(x)
        z_mean, z_log_var = self.z[0], rm.relu(self.z[1])
        e = np.random.randn(len(x), latent_dimension) * sigma
        z_new = z_mean + rm.exp(z_log_var/2)*e
        self.d = dec(z_new)
        d_mean, d_log_var = self.d[0], rm.relu(self.d[1])
        nb, zd = z_log_var.shape
        kl_loss = rm.Variable(0)
        pxz_loss = rm.Variable(0)
        for i in range(nb):
            kl_loss += -0.5*rm.sum(1 + z_log_var[i] - z_mean[i]**2 - rm.exp(z_log_var[i]))
            pxz_loss += rm.sum(0.5*d_log_var[i] + (x[i]-d_mean[i])**2/(2*rm.exp(d_log_var[i])))
        #pxz_loss = rm.sum(0.5*d_log_var + (x-d_mean)**2/(2*rm.exp(d_log_var)))
        vae_loss = (kl_loss + pxz_loss)/nb
        print(dec.output._layers[0].params['w'])
        print(dec.output._layers[0].params['b'])
        print(enc.output._layers[0].params['w'])
        print(enc.output._layers[0].params['b'])
        #pprint(dec.hidden._layers[-1].params['b'])
        return vae_loss 

latent_dimension = 1 
sigma = 1.
enc = EncoderDecoder(2, (2, latent_dimension), units=4, depth=2)
dec = EncoderDecoder(latent_dimension, (2, 2), units=4, depth=2)
vae = Vae(enc, dec)

optimizer = rm.Adam()#Sgd(lr=0.3, momentum=0.4)
plt.clf()
epoch_splits = 10
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(16, epoch_splits*8))
if batch_size == 'Full':
    batch_size = len(train_x)

curve = []
neighbor_period = []
for e in range(epoch):
    s = time()
    perm = np.random.permutation(len(train_x))
    batch_loss = []
    for i in range(0, len(train_x), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = train_x[idx]
        batch_y = train_y[idx]
        with vae.train():
            vae_loss = vae(np.c_[batch_x, batch_y])
        grad = vae_loss.grad()
        grad.update(optimizer)
        batch_loss.append(vae_loss.as_ndarray())
    neighbor_period.append(time()-s)
    test_loss = vae(np.c_[test_x, test_y])
    curve.append([np.array(batch_loss).mean(),test_loss.as_ndarray()])
    if e % epoch_period == epoch_period - 1 or e == epoch:
        if e > epoch//5:
            vae_learn = False
        current_period =  np.array(neighbor_period).mean()
        neighbor_period = []
        ax_ = ax[e//epoch_period]
        curve_na = np.array(curve)
        ax_[0].text(0,0.5, '{:.2f}sec @ epoch'.format(current_period))
        ax_[0].plot(curve_na[:,0])
        ax_[0].plot(curve_na[:,1])
        ax_[0].set_ylim(0,3)
        ax_[0].set_xlim(-2, epoch+10)
        ax_[0].grid()
        vae(np.c_[train_x, train_y])
        pred_train = vae.d[0]
        vae(np.c_[test_x, test_y])
        pred_test = vae.d[0]
        ax_[1].plot(x_axis_org, base, 'k-')
        ax_[1].scatter(x_axis, y_axis, marker='+')
        ax_[1].scatter(pred_train[:,0], pred_train[:,1], c='g', alpha=0.3)
        ax_[1].scatter(pred_test[:,0], pred_test[:,1], c='r', alpha=0.6)
        ax_[1].grid()
        plt.pause(0.5)
fig.savefig('result/vae{}.png'.format(epoch))
plt.pause(3)
