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
from vae_func import VGG_Enc, Dec, Vae2d
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation


def load(fname, shape, offset):
    with gzip.open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape)
    return data

if 0:
    fls = glob('mnist/*.gz')
    data = [[[],[]], [[],[]]]
    for fname in fls:
        shapes = [(-1, 1), (-1, 1, 28, 28)]
        offsets = (8, 16)
        type_idx = 0 if 'labels' in fname else 1
        idx = 0 if 'train' in fname else 1
        data[idx][type_idx] = load(fname, shapes[type_idx], offsets[type_idx])
    data = np.array(data)
    np.save('mnist/data.npy', data)
else:
    data = np.load('mnist/data.npy')

y_train = data[0][0].astype('float32')/255.
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0].astype('float32')/255.
x_test = data[1][1].astype('float32')/255.


set_cuda_active(True)
seed(10)

enc = VGG_Enc()
dec = Dec(latent_dim = 2)
vae = Vae2d(enc, dec)
loss = vae(x_train[:10])

optimizer = rm.Adam()

epoch = 2 
batch_size = 256
N = len(x_train)
curve = []
for e in range(epoch):
    perm = permutation(N)
    batch_loss = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        with vae.train():
            grad = vae(x_train[idx]).grad()
        s = time() - s
        grad.update(optimizer)
        batch_loss.append([
            vae.kl_loss.as_ndarray()[0], 
            vae.recon_loss.as_ndarray(), s])
        loss_na = np.array(batch_loss)
        kl_loss = loss_na[:,0].mean()
        recon_loss = loss_na[:,1].mean()
        s_mean = loss_na[:,2].mean()
        print('{}/{} KL:{:.3f} ReconE:{:.3f} ETA:{:.1f}sec'.format(
            offset, N, kl_loss, recon_loss, (N-offset)/batch_size*s_mean),
            flush=True, end='\r')
    curve.append([kl_loss, recon_loss])
    print('#{} KL:{:.3f} ReconE:{:.3f} @ {:.1f}sec'.format(
        e, kl_loss, recon_loss, loss_na[:,2].sum()))
if latent_dim == 2:
    z_mean, z_log_var = enc(x_train[perm[:100]])
    lft, rgt = z_mean[:,0].min(), z_mean[:,0].max()
    lwr, upr = z_mean[:,1].min(), z_mean[:,1].max()
    # 10 x 10 
    cv = np.zeros((10*28, 10*28))
    h = np.linspace(lft, rgt, 10)
    v = np.linspace(lwr, upr, 10)
    for i in range(10):
        for j in range(10):
            cv[i*28:(i+1)*28, j*28:(j+1)*28] = dec(
                np.array([h[i],v[j]]).reshape(1, 2)
            ).reshape(28, 28)
    cv *= 255
    cv = cv.dtype('uint8')
    io.imshow(cv)
    io.imsave('result/decode.png')