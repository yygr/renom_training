import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from vae_func import EncoderDecoder, Vae, DenseNet

np.set_printoptions(precision=3)

random.seed(10)

N = 100
noise_rate = 0.3
x_noise_rate = 0.1
epoch = 100
batch_size = 'Full'
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

latent_dimension = 1 
if 0:
    enc = EncoderDecoder(2, (2, latent_dimension), 
        units=100, depth=5, batch_normal=True, dropout=True)
    dec = EncoderDecoder(latent_dimension, (2, 2), 
        units=100, depth=5, batch_normal=True, dropout = True)
else:
    enc = DenseNet(2, (2, latent_dimension),
        units=6, growth_rate=12, depth=6)
    dec = DenseNet(latent_dimension, (2, 2),
        units=6, growth_rate=12, depth=6)
vae = Vae(enc, dec)

optimizer = rm.Adam()#Sgd(lr=0.01, momentum=0.)
plt.clf()
epoch_splits = 5
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 3, 
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
        lowest = curve_na.min()
        ax_[0].text(1,lowest, '{:.2f}sec @ epoch'.format(current_period))
        ax_[0].plot(curve_na[:,0])
        ax_[0].plot(curve_na[:,1])
        ax_[0].set_xlim(-2, epoch+10)
        ax_[0].grid()

        check_data = np.c_[x_axis, y_axis]
        if latent_dimension == 1:
            vae(check_data[train_idx])
            ax_[1].scatter(x_axis[train_idx], vae.z_mean, marker='.', alpha=0.7, c='k')
            vae(check_data[test_idx])
            ax_[1].scatter(x_axis[test_idx], vae.z_mean, marker='.', alpha=0.7, c='b')
        else:
            vae(check_data[train_idx])
            ax_[1].scatter(vae.z_mean[:,0], vae.z_mean[:,1],
                marker='.', alpha=0.7, c='k')
            vae(check_data[test_idx])
            ax_[1].scatter(vae.z_mean[:,0], vae.z_mean[:,1],
                marker='.', alpha=0.7, c='b')
        ax_[1].grid()

        vae(np.c_[train_x, train_y])
        pred_train_m, pred_train_d = vae.d_mean, vae.d_log_var
        vae(np.c_[test_x, test_y])
        pred_test_m, pred_test_d = vae.d_mean, vae.d_log_var
        ax_[2].plot(x_axis_org, base, 'k-')
        xerr = np.exp(pred_train_d[:,0]/2)
        yerr = np.exp(pred_train_d[:,1]/2)
        ax_[2].errorbar(pred_train_m[:,0], pred_train_m[:,1], 
            xerr=xerr, yerr=yerr, 
            ecolor='skyblue', c='g', alpha=0.6, fmt='.')
        xerr = np.exp(pred_test_d[:,0]/2)
        yerr = np.exp(pred_test_d[:,1]/2)
        ax_[2].errorbar(pred_test_m[:,0], pred_test_m[:,1],
            xerr=xerr, yerr=yerr, 
            ecolor='skyblue', c='r', alpha=0.6, fmt='.')
        ax_[2].scatter(train_x, train_y, marker='x', alpha=0.7, c='k')
        ax_[2].scatter(test_x, test_y, marker='x', alpha=0.7, c='gold')
        ax_[2].grid()
        plt.pause(0.5)
fig.savefig('result/sin_vae{}.png'.format(epoch))
plt.pause(3)