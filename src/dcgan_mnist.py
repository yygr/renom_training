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
from dcgan_func import Gen, Dis, DCGAN
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation
from sklearn.metrics import confusion_matrix, classification_report


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

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.

x_train = x_train * 2 - 1
x_test = x_test * 2 - 1

set_cuda_active(True)
seed(10)

latent_dim = 200
epoch = 30
batch_size = 256

gen = Gen(latent_dim=latent_dim, batch_normal=True)
dis = Dis()
dcgan = DCGAN(gen, dis)

GAN_dis_opt = rm.Adam()
GAN_gen_opt = rm.Adam()

N = len(x_train)
curve = []
for e in range(epoch):
    perm = permutation(N)
    batch_loss = []
    real = []
    fake = []
    for b, offset in enumerate(range(0, N, batch_size)):
        idx = perm[offset: offset+batch_size]
        s = time()
        with dcgan.train():
            l = dcgan(x_train[idx])
        s = time() - s
        real_cost = dcgan.real_cost.as_ndarray()[0]
        fake_cost = dcgan.fake_cost.as_ndarray()[0]
        gan_loss = l.as_ndarray()[0]
        gen_loss = dcgan.gen_loss.as_ndarray()[0]
        real.append(real_cost)
        fake.append(fake_cost)
        real_mean = np.array(real).mean()
        fake_mean = np.array(fake).mean()
        code = '#'
        rate = 1
        if b % rate == rate - 1:
            code = '^'
            with dcgan.gen.prevent_update():
                l = dcgan.GAN_loss
                l.grad(detach_graph=False).update(GAN_dis_opt)
            with dcgan.dis.prevent_update():
                l = dcgan.gen_loss
                l.grad().update(GAN_gen_opt)
        else:
            code = '/'
            with dcgan.gen.prevent_update():
                l = dcgan.GAN_loss
                l.grad().update(GAN_dis_opt)
        batch_loss.append([gan_loss, gen_loss, real_cost, fake_cost, s])
        loss_na = np.array(batch_loss)
        gan_loss_ = loss_na[:,0].mean()
        gen_loss_ = loss_na[:,1].mean()
        real_cost_ = loss_na[:,2].mean()
        fake_cost_ = loss_na[:,3].mean()
        s_mean = loss_na[:,-1].mean()
        print('{:05}/{} {} GAN:{:.3f} Gen:{:.3f} Real:{:.3f} Fake:{:.3f} ETA:{:.1f}sec'.format(
            offset, N, code, gan_loss_, gen_loss_, real_cost, fake_cost, 
            (N-offset)/batch_size*s_mean), flush=True, end='\r')
    curve.append([gan_loss, gen_loss])
    print('#{:>12} GAN:{:.3f} Gen:{:.3f} Real:{:.3f} Fake:{:.3f} @ {:.1f}sec {:>10}'.format(
        e, gan_loss, gen_loss_, real_cost_, fake_cost_, loss_na[:,-1].sum(), ''))
    res_dim = 16
    cv = np.zeros((res_dim*28, res_dim*28))
    dcgan(x_train[:batch_size])
    ims = dcgan.xp.as_ndarray().reshape(int(res_dim**2), 28, 28)
    for i in range(res_dim):
        for j in range(res_dim):
            cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[i*res_dim+j]
    cv -= cv.min()
    cv /= cv.max()
    cv *= 255
    cv = cv.astype('uint8')
    io.imsave('result/dcgan_fake{}.png'.format(e), cv)
    io.imsave('result/dcgan_fake.png', cv)
