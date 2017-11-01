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
from renom.utility.initializer import Uniform

data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
#$x_train = x_train * 2 - 1
x_test =  x_test.reshape(-1, 28*28)
#x_test = x_test * 2 - 1

latent_dim = 2
epoch = 100
batch_size = 256
model = 'AAE'
e_freq = epoch // 10

class AAE(rm.Model):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.enc = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(200), rm.Relu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.Relu(),
            rm.Dense(latent_dim)
        ])
        self.dec = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(10), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(28*28), rm.Sigmoid()
        ])
        self.dis = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(200), rm.LeakyRelu(),
            rm.Dense(1), rm.Sigmoid()
        ])
    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.z_mu = self.enc(x)
        self.recon = self.dec(self.z_mu)
        self.reconE = rm.mean_squared_error(self.recon, x)

        if 1:
            self.pz = np.random.randn(nb, self.latent_dim)
        else:
            size = nb, self.latent_dim
            self.pz = np.random.uniform(low=-1, high=1, size=size)
        self.Dispz = self.dis(self.pz)
        self.Dispzx = self.dis(self.z_mu)
        self.real = -rm.sum(rm.log(
            self.Dispz + eps
        ))/nb
        self.fake = -rm.sum(rm.log(
            1 - self.Dispzx + eps
        ))/nb
        self.enc_loss = -rm.sum(rm.log(
            self.Dispzx + eps
        ))/nb 
        self.gan_loss = self.real + self.fake
        return self.recon

aae = AAE(latent_dim=latent_dim)

gan_opt = rm.Adam()
enc_opt = rm.Adam()

N = len(x_train)
curve = []
for e in range(epoch):
    perm = permutation(N)
    batch_loss = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        with aae.train():
            aae(x_train[idx])
        with aae.enc.prevent_update():
            l = aae.gan_loss
            l.grad(detach_graph=False).update(gan_opt)
        with aae.dis.prevent_update():
            l = aae.enc_loss + aae.reconE
            l.grad().update(enc_opt)
        s = time() - s
        batch_loss.append([
            aae.gan_loss, 
            aae.enc_loss,
            aae.reconE, 
            aae.real, aae.fake, s])
        loss_na = np.array(batch_loss)
        gan_loss = loss_na[:,0].mean()
        enc_loss = loss_na[:,1].mean()
        reconE = loss_na[:,2].mean()
        real = loss_na[:, 3].mean()
        fake = loss_na[:, 4].mean()
        s_mean = loss_na[:,-1].mean()
        print('{}/{} GAN:{:.3f} Enc:{:.3f} Dec:{:.3f} {:.2f}/{:.2f} ETA:{:.1f}sec'.format(
            offset, N, gan_loss, enc_loss, reconE, 
            real, fake,
            (N-offset)/batch_size*s_mean),
            flush=True, end='\r')
    curve.append([gan_loss, reconE])
    print('#{} GAN:{:.3f} Enc:{:.3f} Dec:{:.3f} {:.2f}/{:.2f} @ {:.1f}sec {:>10}'.format(
        e, gan_loss, enc_loss, reconE, real, fake, loss_na[:,-1].sum(), ''))

    if e % e_freq != e_freq - 1:
        continue

    res = aae.enc(x_test[:batch_size])
    res = res.as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = aae.enc(x_test[i:i+batch_size])
        res = np.r_[res, z_mean.as_ndarray()]
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim > 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.scatter(res[:,0], res[:,1], c=y_test, alpha=0.7, marker='.')
        plt.savefig('result/AAE_latent{}_{}.png'.format(latent_dim, e))
        plt.savefig('result/AAE_latent{}.png'.format(latent_dim))

    res_dim = 16
    ims = aae(x_test[:batch_size])
    ims = ims.reshape(-1, 28, 28)
    cv = np.zeros((res_dim*28, res_dim*28))
    for i in range(res_dim):
        for j in range(res_dim):
            idx = i*res_dim+j
            cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[idx]
    cv -= cv.min()
    cv /= cv.max()
    cv *= 255
    cv = cv.astype('uint8')
    io.imsave('result/AAE_decode{}.png'.format(e), cv)
    io.imsave('result/AAE_decode.png', cv)

    if latent_dim == 2:
        v = np.linspace(1, -1, res_dim)
        h = np.linspace(-1, 1, res_dim)
        data = np.zeros((res_dim**2, latent_dim))
        for i in range(res_dim):
            for j in range(res_dim):
                data[i*res_dim+j,0] = v[i]
                data[i*res_dim+j,1] = h[j]
        ims = aae.dec(data).reshape(-1, 28, 28)
        cv = np.zeros((res_dim*28, res_dim*28))
        for i in range(res_dim):
            for j in range(res_dim):
                idx = i*res_dim+j
                cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[idx]
        cv -= cv.min()
        cv /= cv.max()
        cv *= 255
        cv = cv.astype('uint8')
        io.imsave('result/AAE_map{}.png'.format(e), cv)
        io.imsave('result/AAE_map.png', cv)