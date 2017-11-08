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
from aae_func import AAE
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from skimage import io, color

seed(10)

data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
#x_train = x_train * 2 - 1
#x_test = x_test * 2 - 1

def one_hot(data, size=11):
    temp = np.zeros((len(data), size))
    temp[np.arange(len(data)),data.reshape(-1)]=1
    return temp

y_train_1 = one_hot(y_train)
idx = np.random.permutation(len(y_train))[:50000]
y_train_1[idx] = np.r_[np.zeros(10),np.ones(1)].reshape(1,11)
y_test_ = one_hot(y_test)

latent_dim = 2
epoch = 30
batch_size = 256
shot_freq = epoch//10

train = True

#ae = AAE(latent_dim, prior='uniform')
if 1:
    ae = AAE(latent_dim, hidden=100, 
        prior='10d-gaussians', mode='incorp_label', label_dim=10)
    #ae = AAE(latent_dim, prior='gaussians')
else:
    im = io.imread('src/guri.png')
    im = color.rgb2gray(im)
    v, h = np.where(im==im.max())
    v = -v/500 + 1.
    h = h/500 - 1.
    dist = np.c_[h, v]
    ae = AAE(latent_dim, prior='predist', prior_dist=dist)

"""
dis_fm_opt = rm.Adam()
dis_mc_opt = rm.Adam()
enc_fm_opt = rm.Adam()
enc_mc_opt = rm.Adam()
"""
dis_opt = rm.Adam(lr=0.0001, b=0.9)
enc_opt = rm.Adam(lr=0.0001, b=0.9)
dec_opt = rm.Adam()

N = len(x_train)
curve = []
for e in range(epoch):
    if not train:
        continue
    """
    if e == 10:
        dis_opt = rm.Adam(lr=0.00001)
        enc_opt = rm.Adam(lr=0.00001)
    elif e == 100:
        dis_opt = rm.Adam(lr=0.000001)
        enc_opt = rm.Adam(lr=0.000001)
    """
    perm = permutation(N)
    batch_loss = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        train_data = x_train[idx]# + random.randn(len(idx),28*28)*0.3
        with ae.train():
            ae(train_data, y=y_train_1[idx])
        with ae.enc.prevent_update():
            l = ae.gan_loss
            l.grad(detach_graph=False).update(dis_opt)
        with ae.dis.prevent_update():#, ae.dec.prevent_update():
            l = ae.enc_loss + ae.reconE
            l.grad().update(enc_opt)
            #l.grad(detach_graph=False).update(enc_opt)
        """
        with ae.enc.prevent_update():
            l = ae.reconE
            l.grad().update(dec_opt)
        """
        s = time() - s
        batch_loss.append([
            ae.gan_loss, ae.enc_loss, ae.reconE, 
            ae.real_count, ae.fake_count, s])
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
        print(print_str, flush=True, end='\r')
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
        gan_loss, enc_loss, recon_loss
    )
    print_str += ' {:.2f}/{:.2f}'.format(real, fake)
    print_str += ' @ {:.1f} sec {:>20}'.format(
        loss_na[:,-1].sum(), ''
    )
    print(print_str)

    if e % shot_freq != shot_freq - 1:
        continue
    res = ae.enc(x_test[:batch_size])
    res = res
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = ae.enc(x_test[i:i+batch_size])
        res = np.r_[res, z_mean]
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.figure(figsize=(7,7))
        plt.scatter(res[:,0], res[:,1], c=y_test.reshape(-1), alpha=0.5)
        plt.axis('equal')
        plt.grid()
        plt.savefig('result/AAEi_latent{}_{}.png'.format(latent_dim, e))
        plt.savefig('result/AAEi_latent.png')

    res_dim = 16
    ims = ae(x_test[:batch_size], y_test_[:batch_size])
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
    io.imsave('result/AAEi{}_decode{}.png'.format(latent_dim, e), cv)
    io.imsave('result/AAEi_decode.png', cv)
"""
if train:
    enc.save('model/enc.h5')
    dec.save('model/dec.h5')
else:
    enc.load('model/enc.h5')
    dec.load('model/dec.h5')
"""
