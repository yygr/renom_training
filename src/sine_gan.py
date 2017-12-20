import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pdb import set_trace
from glob import glob
from re import search
from skimage import io
from os import listdir, makedirs, path, sep
from gan_func import GAN
from numpy import random
from renom.cuda.cuda import set_cuda_active
from time import time
from renom.utility.initializer import Uniform
from subprocess import call

random.seed(10)
set_cuda_active(True)

series = 100
scale = 10.
n = 5000
x = np.linspace(-np.pi, np.pi, series)
_y = np.sin(x) * scale
y = [_y + np.random.randn(series)*(scale/100) for _ in range(n)]
y = np.array(y).astype('float32')
perm = np.random.permutation(n)
x_train = y[perm[100:]]
x_test = y[perm[:100]]

hidden = 1000
gan_dim = 1000
batch_size = 100
epoch = 100
shot_period = 10
shot_freq = epoch // shot_period
s = batch_size, series
gen = rm.Sequential([
    #rm.Dense(hidden), 
    #rm.BatchNormalize(),
    rm.Dense(hidden), rm.Tanh(),
    rm.Dense(series)
])

dis = rm.Sequential([
    #rm.Dense(hidden), #rm.LeakyRelu(),
    #rm.BatchNormalize(),
    rm.Dense(hidden), #rm.LeakyRelu(),
    rm.Dense(1), rm.Sigmoid()
])


gan = GAN(gen, dis, latent_dim=gan_dim)
initial_lr = 0.001
last_lr = initial_lr/1000
_b = .5 # default is .9
_c = 0.5 # center
_m = 0.2 # margin
down_rate = 1.02
gen_opt = rm.Adam(lr=initial_lr, b=_b)
dis_opt = rm.Adam(lr=initial_lr, b=_b)
gen_lr = np.linspace(initial_lr, last_lr, epoch // shot_period + 1)
dis_lr = np.linspace(initial_lr, last_lr, epoch // shot_period + 1)

N = len(x_train)
batch_shape = batch_size, series
history = []
_train = np.zeros(batch_shape)
for e in range(epoch):
    code = '|'
    if 0:#e % shot_freq == shot_freq - 1:
        _gen_lr = gen_lr[e//shot_freq]
        _dis_lr = dis_lr[e//shot_freq]
        print("###{}/{}###".format(_gen_lr, _dis_lr))
        gen_opt._lr = _gen_lr
        dis_opt._lr = _dis_lr
    perm = random.permutation(N)
    batch_history = []
    for offset in range(0, len(x_train), batch_size):
        batch_start = time()
        temp_idx = perm[offset:offset+batch_size]
        _train[:len(temp_idx)] = x_train[temp_idx]
        with gan.train():
            gan(_train)#, noise=0.5)
        forwarded_time = time()
        gan_train, gen_train = True, True
        if offset > batch_size:
            _mean = np.array(batch_history).mean(0)
            real = float(_mean[4])
            fake = float(_mean[5])
            gan_train = False if _c-_m < real < _c+_m else True 
            gen_train = False if _c-_m < fake < _c+_m else True 
            if gen_train == False and gan_train == False:
                gen_opt._lr /= down_rate
                dis_opt._lr /= down_rate
                _m /= down_rate
                code = '#'
        if gan_train:
            with gan.gen.prevent_update():
                l = gan.GAN_loss
                l.grad(detach_graph=False).update(dis_opt) 
        if gen_train:
            with gan.dis.prevent_update():
                l = gan.gen_loss
                l.grad().update(gen_opt)
        end_time = time()
        batch_history.append([
            float(gan.GAN_loss.as_ndarray()),
            float(gan.gen_loss.as_ndarray()),
            float(gan.real_cost.as_ndarray()),
            float(gan.fake_cost.as_ndarray()),
            float(gan.real_count),
            float(gan.fake_count),
            forwarded_time - batch_start,
            end_time - forwarded_time,
            end_time - batch_start
        ])
        mean = np.array(batch_history).mean(0)
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' GAN:{:.2f}/{:.2f}={:.2f}/{:.2f}'.format(
            mean[0], mean[1], mean[2], mean[3]
        )
        print_str += '={:.2f}/{:.2f}'.format(
            mean[4], mean[5]
        )
        print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*mean[-1])
        print_str += ' {:.2f},{:.2f} {:>10}'.format(
            mean[-2], mean[-1],'')
        print_str += '{:.2e}{}{:.2e}'.format(_m, code, dis_opt._lr)
        print(print_str, flush=True, end='\r')
    history.append(np.r_[mean, np.array(batch_history)[:,-1].sum()])
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' GAN:{:.2f}/{:.2f}={:.2f}/{:.2f}'.format(
        mean[0], mean[1], mean[2], mean[3]
    )
    print_str += '={:.2f}/{:.2f}'.format(
        mean[4], mean[5]
    )
    print_str += ' @ {:.1f} sec {:>20}'.format(
        np.array(history)[-1,-1], ''
    )
    print(print_str)

    if e % shot_freq == shot_freq - 1 or e == epoch - 1:
        gan(x_test[:batch_size])
        res = gan.xz.as_ndarray()
        dispz = gan.Disxz.as_ndarray()

        plt.clf()
        org = x_test[:batch_size]
        for j in range(batch_size):
            plt.plot(res[j], 
                alpha=dispz[j], 
                lw=.1, c='b')
        #plt.plot(org[:batch_size].mean(0), c='k', lw=1)
        plt.plot(_y, c='r', lw=2)
        plt.plot(res[np.argmax(dispz)], c='k', lw=2)
        plt.text(0,5,'{}'.format(e+1))
        plt.grid()
        figname = 'result/sine{}gan.png'.format(e+1)
        plt.savefig(figname)
        call('cp {} result/gan_gan.png'.format(figname), shell=True)
print('finished @ {:.1f} sec'.format(np.array(history)[:,-1].sum()))
