# encoding:utf-8
import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from skimage import io
from glob import glob
import gzip
from vae_func import VGG_Enc, Dec, Vae2d, keras_Enc, keras_Dec, Densenet_Enc
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation
from sklearn.manifold import TSNE

data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
#x_train = x_train * 2 - 1
x_test =  x_test.reshape(-1, 28*28)
#x_test = x_test * 2 - 1

latent_dim = 50
epoch = 30 
batch_size = 256
model = 'AE'

class AE(rm.Model):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.enc = rm.Sequential([
            #rm.BatchNormalize(),
            #rm.Dense(100),
            #rm.Dropout(),
            #rm.Relu(),
            rm.Dense(latent_dim), rm.Tanh()
        ])
        self.dec = rm.Sequential([
            #rm.BatchNormalize(),
            #rm.Dense(100),
            #rm.Dropout(),
            #rm.LeakyRelu(),
            rm.Dense(28*28), rm.Sigmoid()
        ])
    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.z_mu = self.enc(x)
        self.decd = self.dec(self.z_mu)
        self.reconE = rm.mean_squared_error(self.decd, x)
        return self.decd

ae = AE(latent_dim=latent_dim)

ae_opt = rm.Adam()

N = len(x_train)
curve = []
for e in range(epoch):
    perm = permutation(N)
    batch_loss = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        with ae.train():
            ae(x_train[idx])
            l = ae.reconE
        s = time() - s
        l.grad().update(ae_opt)
        batch_loss.append([l, s])
        loss_na = np.array(batch_loss)
        reconE = loss_na[:,0].mean()
        s_mean = loss_na[:,-1].mean()
        print('{}/{} ReconE:{:.3f} ETA:{:.1f}sec'.format(
            offset, N, reconE,
            (N-offset)/batch_size*s_mean),
            flush=True, end='\r')
    curve.append([reconE, reconE])
    print('#{} ReconE:{:.3f} @ {:.1f}sec {:>10}'.format(
        e, reconE, loss_na[:,-1].sum(), ''))

    res = ae.enc(x_test[:batch_size])
    res = res.as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = ae.enc(x_test[i:i+batch_size])
        res = np.r_[res, z_mean]
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.scatter(res[:,0], res[:,1], c=y_test, alpha=0.5)
        plt.savefig('result/_AE_latent{}.png'.format(e))
        plt.savefig('result/_AE_latent.png')

    res_dim = 16
    ims = ae(x_test[:batch_size])
    ims = ims.reshape(-1, 28, 28)
    #og = x_test[:batch_size].reshape(-1, 28, 28)
    cv = np.zeros((res_dim*28, res_dim*28))
    for i in range(res_dim):
        for j in range(res_dim):
            idx = i*res_dim+j
            cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[idx] #+ og[idx]
    cv -= cv.min()
    cv /= cv.max()
    cv *= 255
    cv = cv.astype('uint8')
    io.imsave('result/_AE_decode{}.png'.format(e), cv)
    io.imsave('result/_AE_decode.png', cv)