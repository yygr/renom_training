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

data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
x_train = x_train * 2 - 1
x_test =  x_test.reshape(-1, 28*28)
x_test = x_test * 2 - 1

latent_dim = 100
epoch = 100 
batch_size = 256
model = 'AAE'

class AAE(rm.Model):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.enc = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(200), rm.Relu(),
            rm.Dropout(),
            #rm.BatchNormalize(),
            #rm.Dense(100), rm.Relu(),
            #rm.Dropout()
        ])
        self.zm = rm.Dense(latent_dim)
        self.zlv = rm.Dense(latent_dim)
        self.dec = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(400), rm.LeakyRelu(),
            rm.Dropout(),
            rm.Dense(28*28), rm.Tanh()
        ])
        self.dis = rm.Sequential([
            #rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            #rm.Dropout(),
            #rm.Dense(100), rm.LeakyRelu(),
            #rm.Dropout(),
            rm.Dense(1), rm.Sigmoid()
        ])
    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.pzx = self.enc(x)
        self.z_mu = self.zm(self.pzx)
        self.z_lvar = self.zlv(self.pzx)
        self.decd = self.dec(self.z_mu)
        self.reconE = rm.mean_squared_error(x, self.decd)
        self.pz = np.random.randn(nb, self.latent_dim)
        self.Dispz = self.dis(self.pz)
        self.Dispzx = self.dis(self.z_mu)
        self.real = -rm.sum(rm.log(
            self.Dispz + eps
        ))/nb
        self.fake = -rm.sum(rm.log(
            1 - self.Dispzx + eps
        ))/nb
        self.fake_ = -rm.sum(rm.log(
            self.Dispzx + eps
        ))
        self.gan_loss = self.real + self.fake
        self.ae_loss = self.reconE + self.fake_
        return self.reconE + self.gan_loss

aae = AAE(latent_dim=latent_dim)

gan_opt = rm.Adam()
dec_opt = rm.Adam()
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
        s = time() - s
        #with aae.enc.prevent_update():
        l = aae.reconE
        l.grad(detach_graph=False).update(dec_opt)
        with aae.enc.prevent_update():
            l = aae.gan_loss
            l.grad(detach_graph=False).update(gan_opt)
        """
        with aae.dis.prevent_update():
            l = aae.fake_
            l.grad().update(enc_opt)
        """
        batch_loss.append([
            aae.gan_loss, 
            aae.reconE, 
            aae.real, aae.fake, s])
        loss_na = np.array(batch_loss)
        gan_loss = loss_na[:,0].mean()
        reconE = loss_na[:,1].mean()
        real = loss_na[:, 2].mean()
        fake = loss_na[:, 3].mean()
        s_mean = loss_na[:,-1].mean()
        print('{}/{} GAN:{:.3f} ReconE:{:.3f} {:.2f}/{:.2f} ETA:{:.1f}sec'.format(
            offset, N, gan_loss, reconE, 
            real, fake,
            (N-offset)/batch_size*s_mean),
            flush=True, end='\r')
    curve.append([gan_loss, reconE])
    print('#{} GAN:{:.3f} ReconE:{:.3f} {:.2f}/{:.2f} @ {:.1f}sec {:>10}'.format(
        e, gan_loss, reconE, real, fake, loss_na[:,-1].sum(), ''))

    res = aae.zm(aae.enc(x_test[:batch_size]))
    res = res.as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = aae.zm(aae.enc(x_test[i:i+batch_size]))
        res = np.r_[res, z_mean.as_ndarray()]
    if latent_dim == 2:
        plt.clf()
        plt.scatter(res[:,0], res[:,1], c=y_test)
        plt.savefig('result/AAE_latent{}.png'.format(e))
        plt.savefig('result/AAE_latent.png')

    res_dim = 16
    ims = aae.dec(res[:batch_size])
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