# encoding:utf-8
import gzip
from glob import glob
from pdb import set_trace
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.random import permutation, seed
from skimage import color, io
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

import renom as rm
from aae_func import AAE
from renom.cuda.cuda import set_cuda_active
from renom.utility.initializer import Gaussian, Uniform

# --- configuration ---
seed(10)
set_cuda_active(True) # gpu is mandatory
latent_dim = 2
epoch = 10
batch_size = 256
shot_freq = epoch//10
hidden = 1000
train = True

# --- data loading & prepairing ---
data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

def one_hot(data, size=11):
    temp = np.zeros((len(data), size))
    temp[np.arange(len(data)),data.reshape(-1)]=1
    return temp

y_train_1 = one_hot(y_train)
idx = random.permutation(len(y_train))[10000:]
y_train_1[idx] = np.r_[np.zeros(10),np.ones(1)].reshape(1,11)

# --- model configuration ---

enc_base = rm.Sequential([
    #rm.BatchNormalize(),
    rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
    rm.BatchNormalize(),
    rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
    # xxx rm.BatchNormalize(),
    # Genの最後にBNを配置してはだめ
    rm.Dense(latent_dim, initializer=Uniform())
])
dec = rm.Sequential([
    #rm.BatchNormalize(),
    rm.Dense(hidden), rm.LeakyRelu(),
    rm.BatchNormalize(),
    rm.Dense(hidden), rm.LeakyRelu(),
    #rm.BatchNormalize(),
    rm.Dense(28*28), rm.Sigmoid()
])

if 1:
    ae = AAE(
        enc_base, dec, batch_size,
        latent_dim = latent_dim, hidden=200, 
        prior='10d-gaussians', mode='incorp_label', label_dim=10)
    #ae = AAE(latent_dim, prior='gaussians')
elif 0:
    ae = AAE(latent_dim, hidden=100,
        prior='normal')
else:
    im = io.imread('src/guri.png')
    im = color.rgb2gray(im)
    v, h = np.where(im==im.max())
    v = -v/500 + 1.
    h = h/500 - 1.
    dist = np.c_[h, v]
    ae = AAE(latent_dim, prior='predist', prior_dist=dist)

dis_opt = rm.Adam(lr=0.001, b=0.5)
enc_opt = rm.Adam(lr=0.001, b=0.5)

N = len(x_train)
curve = []
for e in range(epoch):
    if not train:
        continue
    perm = permutation(N)
    batch_history = []
    k = 1 
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        train_data = x_train[idx]
        with ae.train():
            ae(train_data, y=y_train_1[idx])
        with ae.enc.prevent_update():
            l = ae.gan_loss
            l.grad(detach_graph=False).update(dis_opt)
        if e % k == 0:
            with ae.dis.prevent_update():
                # Force reconstruction error to small
                # for assigning latent location
                l = ae.enc_loss + 0.1*ae.reconE
                l.grad().update(enc_opt)
        s = time() - s
        batch_history.append([
            float(ae.gan_loss.as_ndarray()), 
            float(ae.enc_loss.as_ndarray()), 
            float(ae.reconE.as_ndarray()), 
            float(ae.real_count), 
            float(ae.fake_count), s])
        mean = np.array(batch_history).mean(0)
        """
        loss_na = np.array(batch_loss)
        gan_loss = loss_na[:,0].mean()
        enc_loss = loss_na[:,1].mean()
        recon_loss = loss_na[:,2].mean()
        real = loss_na[:,3].mean()
        fake = loss_na[:,4].mean()
        s_mean = loss_na[:,-1].mean()
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
            gan_loss, enc_loss, recon_loss
        )
        print_str += ' {:.2f}/{:.2f}'.format(real, fake)
        print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*s_mean)
        """
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
            mean[0], mean[1], mean[2] 
        )
        print_str += ' {:.2f}/{:.2f}'.format(mean[3], mean[4])
        print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*mean[-1])
        print(print_str, flush=True, end='\r')
    """
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
        gan_loss, enc_loss, recon_loss
    )
    print_str += ' {:.2f}/{:.2f}'.format(real, fake)
    print_str += ' @ {:.1f} sec {:>20}'.format(
        loss_na[:,-1].sum(), ''
    )
    """
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
        mean[0], mean[1], mean[2] 
    )
    print_str += ' {:.2f}/{:.2f}'.format(mean[3], mean[4])
    print_str += ' @ {:.1f} sec {:>20}'.format(
        np.array(batch_history)[:,-1].sum(), ''
    )
    print(print_str)


    #if 1:
    # @@@ inference @@@
    #ae.set_models(inference=True)
    res = ae.enc(x_test[:batch_size]).as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = ae.enc(x_test[i:i+batch_size]).as_ndarray()
        res = np.r_[res, z_mean]
    #ae.set_models(inference=False)
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.figure(figsize=(7,7))
        plt.scatter(res[:,0], res[:,1], 
            c=y_test.reshape(-1), alpha=0.5,
            label='{}'.format(e+1))
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        if e % shot_freq == shot_freq - 1:
            plt.savefig('result/AAEi_latent{}_{}.png'.format(latent_dim, e))
        plt.savefig('result/AAEi_latent.png')

    #ae.set_models(inference=True)
    res = ae.enc(x_train[:batch_size]).as_ndarray()
    for i in range(batch_size, len(x_train), batch_size):
        z_mean = ae.enc(x_train[i:i+batch_size]).as_ndarray()
        res = np.r_[res, z_mean]
    #ae.set_models(inference=False)
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.figure(figsize=(7,7))
        plt.scatter(res[:,0], res[:,1], 
            c=y_train.reshape(-1), alpha=0.5,
            label='{}'.format(e+1))
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        if e % shot_freq == shot_freq - 1:
            plt.savefig('result/train_latent{}_{}.png'.format(latent_dim, e))
        plt.savefig('result/train_latent.png')


    res_dim = 16
    ims = ae(x_test[:batch_size]).as_ndarray()
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
    if e % shot_freq == shot_freq - 1:
        io.imsave('result/AAEi{}_decode{}.png'.format(latent_dim, e), cv)
    io.imsave('result/AAEi_decode.png', cv)

    if latent_dim == 2:
        v = np.linspace(5, -5, res_dim)
        h = np.linspace(-5, 5, res_dim)
        data = np.zeros((res_dim**2, latent_dim))
        for i in range(res_dim):
            for j in range(res_dim):
                data[i*res_dim+j,0] = h[j]
                data[i*res_dim+j,1] = v[i]
        ims = ae.dec(data).as_ndarray().reshape(-1, 28, 28)
        cv = np.zeros((res_dim*28, res_dim*28))
        for i in range(res_dim):
            for j in range(res_dim):
                idx = i*res_dim+j
                cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[idx]
        cv -= cv.min()
        cv /= cv.max()
        cv *= 255
        cv = cv.astype('uint8')
        if e % shot_freq == shot_freq - 1:
            io.imsave('result/AAEi_map{}.png'.format(e), cv)
        io.imsave('result/AAEi_map.png', cv)
    #ae.set_models(inference=False)
"""
if train:
    enc.save('model/enc.h5')
    dec.save('model/dec.h5')
else:
    enc.load('model/enc.h5')
    dec.load('model/dec.h5')
"""
