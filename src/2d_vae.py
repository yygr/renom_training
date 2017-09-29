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
epoch = 500
batch_size = 'Full'
vae_learn = True

if 0:
    z_true = np.linspace(0,1,N)
    noise = np.random.randn(N)
    noise = np.abs(noise)
    noise /= noise.max()
    z_noised = z_true + noise
    r = np.power(z_noised,0.5)
    phi = 0.25 * np.pi * z_noised
    x1 = r*np.cos(phi) - 0.5
    x2 = r*np.sin(phi) - 0.5

    r_true = np.power(z_true, 0.5)
    phi_true = 0.25 * np.pi * (z_true)
    x1_true = r_true * np.cos(phi_true) - 0.5
    x2_true = r_true * np.sin(phi_true) - 0.5
    Y = np.transpose(np.reshape((x1_true,x2_true), (2,N)))

    x1 = np.random.normal(x1, 0.1*np.power(z_noised,2), N)
    x2 = np.random.normal(x2, 0.1*np.power(z_noised,2), N)
else:
    z_true = np.random.uniform(0,1,N)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r*np.cos(phi)
    x2 = r*np.sin(phi)

    x1 = np.random.normal(x1, 0.10*np.power(z_true,2), N)
    x2 = np.random.normal(x2, 0.10*np.power(z_true,2), N)
X = np.transpose(np.reshape((x1,x2), (2, N)))
X = np.asarray(X, dtype='float32')
sortedX = X[np.argsort(phi)]
sortedP = np.sort(phi)
if 0:
    plt.scatter(X[:,0],X[:,1], marker='.', alpha=0.2)
    #plt.plot(Y[:,0], Y[:,1], alpha=0.2, c='r')
    plt.show()

idx = random.permutation(N)
x_train = X[idx[::2]] 
x_test = X[idx[1::2]]
latent_dimension = 1
if 0:
    enc = EncoderDecoder(2, (2, latent_dimension), 
        units=6, depth=3)#, batch_normal=True)
    dec = EncoderDecoder(latent_dimension, (2, 2), 
        units=6, depth=3)#, batch_normal=True)
else:
    enc = DenseNet(2, (2, latent_dimension),
        units=6, depth=4)
    dec = DenseNet(latent_dimension, (2, 2),
        units=6, depth=4)
vae = Vae(enc, dec)

optimizer = rm.Adam()#Sgd(lr=0.01, momentum=0.)
plt.clf()
epoch_splits = 5
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 3, 
    figsize=(16, epoch_splits*8))
if batch_size == 'Full':
    batch_size = len(x_train)

curve = []
neighbor_period = []
for e in range(epoch):
    s = time()
    perm = np.random.permutation(len(x_train))
    batch_loss = []
    for i in range(0, len(x_train), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = x_train[idx]
        with vae.train():
            vae_loss = vae(batch_x)
        grad = vae_loss.grad()
        grad.update(optimizer)
        batch_loss.append(vae_loss.as_ndarray())
    neighbor_period.append(time()-s)
    test_loss = vae(x_test)
    curve.append([np.array(batch_loss).mean(),test_loss.as_ndarray()])
    if e % epoch_period == epoch_period - 1 or e == epoch:
        if e > epoch//5:
            vae_learn = False
        current_period =  np.array(neighbor_period).mean()
        neighbor_period = []
        ax_ = ax[e//epoch_period]
        curve_na = np.array(curve)
        ax_[0].text(0,0, '{:.2f}sec @ epoch'.format(current_period))
        ax_[0].plot(curve_na[:,0])
        ax_[0].plot(curve_na[:,1])
        #ax_[0].set_ylim(-3,3)
        ax_[0].set_xlim(-2, epoch+10)
        ax_[0].grid()
        vae(x_train)
        pred_train_m, pred_train_d = vae.d_mean, vae.d_log_var
        train_zm, train_lv = vae.z_mean, vae.z_log_var
        vae(x_test)
        pred_test_m, pred_test_d = vae.d_mean, vae.d_log_var
        test_zm, test_lv = vae.z_mean, vae.z_log_var
        vae(sortedX)
        ax_[1].plot(sortedP, vae.z_mean, marker=',', alpha=0.7, c='r')
        ax_[1].grid()
        ax_[2].scatter(pred_train_m[:,0], pred_train_m[:,1], 
            c= 'g', alpha=0.4, marker='.')
        """
        xerr = np.exp(pred_train_d[:,0]/2)
        yerr = np.exp(pred_train_d[:,1]/2)
        ax_[2].errorbar(pred_train_m[:,0], pred_train_m[:,1], 
            xerr=xerr, yerr=yerr, 
            ecolor='skyblue', c='g', alpha=0.6, fmt='.')
        """
        ax_[2].scatter(pred_test_m[:,0], pred_test_m[:,1], 
            c= 'r', alpha=0.4, marker='.')
        """
        xerr = np.exp(pred_test_d[:,0]/2)
        yerr = np.exp(pred_test_d[:,1]/2)
        ax_[2].errorbar(pred_test_m[:,0], pred_test_m[:,1],
            xerr=xerr, yerr=yerr, 
            ecolor='skyblue', c='r', alpha=0.6, fmt='.')
        """
        ax_[2].scatter(x_test[:,0], x_test[:,1], marker='x', alpha=0.6, c='gold')
        ax_[2].scatter(x_train[:,0], x_train[:,1], marker='x', alpha=0.6, c='k')
        ax_[2].grid()
        plt.pause(0.5)
fig.savefig('result/2d_vae{}.png'.format(epoch))
plt.pause(3)