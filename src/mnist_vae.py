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
from vae_func import Enc, VAE, VGG, Dec2d
from vae_func import VGG, Dec2d 
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation
from os import path, makedirs
from renom.utility.initializer import Uniform


model_id = 'VAE'

# --- prepairing ---
def load(fname, shape, offset):
    with gzip.open(fname, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape)
    return data

data_name = 'mnist/data.npy'
if not path.exists(data_name):
    fls = glob('mnist/*.gz')
    data = [[[],[]], [[],[]]]
    for fname in fls:
        shapes = [(-1, 1), (-1, 1, 28, 28)]
        offsets = (8, 16)
        type_idx = 0 if 'labels' in fname else 1
        idx = 0 if 'train' in fname else 1
        data[idx][type_idx] = load(fname, shapes[type_idx], offsets[type_idx])
    data = np.array(data)
    np.save(data_name, data)
else:
    data = np.load(data_name)

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.

# --- configuration ---
set_cuda_active(True)
seed(10)

latent_dim = 2
epoch = 100
shot_period = epoch/5
batch_size = 256
hidden = 100
# Rate of KL vs ReconE
r = 1. 

model = '' # or 'vgg'
debug = True

# --- network definition ---
if model == 'vgg':
    batch_shape = (batch_size, 1, 28, 28)
    pre=VGG(
        input_shape = batch_shape[1:],
        batch_size = batch_size,
        check_network = debug,
        growth_factor = 1.2
    )
    enc = Enc(pre=pre, latent_dim=latent_dim)
    dec = Dec2d(
        input_params = latent_dim,
        first_shape = pre.output_shape,
        output_shape = batch_shape,
        check_network = debug,
    )    
else: # fully connected network
    batch_shape = (batch_size, 28*28)
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    enc_pre = rm.Sequential([
        rm.Dense(hidden), rm.Relu(),
        rm.BatchNormalize(),
        rm.Dense(hidden, initializer=Uniform()), rm.Relu()
    ])
    enc = Enc(enc_pre, latent_dim)
    dec = rm.Sequential([
        rm.Dense(hidden), rm.Relu(),
        rm.BatchNormalize(),
        rm.Dense(28*28), rm.Sigmoid(),
    ])
vae = VAE(enc, dec, latent_dim)
optimizer = rm.Adam()

N = len(x_train)
history = []
for e in range(epoch):
    perm = permutation(N)
    batch_history = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        s = time()
        tmp = x_train[idx]
        train_data = np.zeros(batch_shape)
        train_data[:len(tmp)] = tmp
        with vae.train():
            vae(train_data)
        forward_time = time() - s
        # backward
        l = vae.kl_loss + r*vae.reconE
        l.grad().update(optimizer)
        s = time() - s

        backward_time = s - forward_time
        batch_history.append([
            float(vae.kl_loss.as_ndarray()), 
            float(vae.reconE.as_ndarray()), 
            forward_time,
            backward_time,
            s
        ])
        mean = np.array(batch_history).mean(0)
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' KL:{:.3f} ReconE:{:.3f} ETA:{:.1f}sec'.format(
            mean[0], mean[1], (N-offset)/batch_size*mean[-1]
        )
        print(print_str, flush=True, end='\r')
    vae.set_models(inference=True)
    decd = vae(x_test[:batch_size]).as_ndarray()
    lvec = vae.enc.zm.as_ndarray()
    kl_ = float(vae.kl_loss.as_ndarray())
    rE_ = float(vae.reconE.as_ndarray())
    for i in range(batch_size, len(x_test), batch_size):
        forward_data = np.zeros(batch_shape)
        tmp = x_test[i:i+batch_size]
        forward_data[:len(tmp)] = tmp
        decd = np.r_[decd, vae(forward_data).as_ndarray()[:len(tmp)]]
        lvec = np.r_[lvec, vae.enc.zm.as_ndarray()[:len(tmp)]]
        kl_ += vae.kl_loss.as_ndarray()
        rE_ += vae.reconE.as_ndarray()
    kl_ /= len(x_test)/batch_size
    rE_ /= len(x_test)/batch_size
    vae.set_models(inference=False)
    history.append(np.r_[mean, np.array([kl_, rE_])])
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' KL:{:.3f} ReconE:{:.3f}'.format(mean[0], mean[1])
    print_str += '|KL:{:.3f} ReconE:{:.3f}'.format(kl_, rE_)
    print_str += ' @ {:.1f} sec {:>20}'.format(
        np.array(batch_history)[:,-1].sum(), ''
    )
    print(print_str)

    if not e%shot_period == shot_period - 1:
        continue
    outdir = 'result/{}'.format(model_id)
    if not path.exists(outdir):
        makedirs(outdir)
    suffix = int(e//shot_period)+1
    print('-'*len(print_str))

    # latent space 
    if latent_dim == 2:
        plt.clf()
        plt.scatter(lvec[:,0], lvec[:,1], 
            c=y_test.reshape(-1,),
            label='{}th - KL:{:.2f}, ReconE:{:.2f}'.format(e+1, kl_, rE_)
        )
        plt.tight_layout()
        plt.legend()
        plt.grid()
        outname = '{}/latent{}.png'.format(outdir, suffix)
        plt.savefig(outname)
        plt.close()

    # reconstruction status
    # 16 x 16 = 256
    res_dim = 16
    def emb_cv(data):
        cv = np.zeros((res_dim*28, res_dim*28))
        for i in range(res_dim):
            for j in range(res_dim):
                cv[i*28:(i+1)*28, j*28:(j+1)*28] = data[
                    i*res_dim + j
                ].reshape(28, 28)
        cv *= 255
        cv = cv.astype('uint8')
        return cv

    cv = emb_cv(decd[:batch_size])
    io.imsave('{}/decoded{}.png'.format(outdir, suffix), cv)

    z_mean = lvec[:batch_size]
    lft, rgt = z_mean[:,0].min(), z_mean[:,0].max()
    lwr, upr = z_mean[:,1].min(), z_mean[:,1].max()
    h = np.linspace(lft, rgt, res_dim)
    v = np.linspace(lwr, upr, res_dim)
    data = []
    for i in range(res_dim):
        for j in range(res_dim):
            data.append(
                np.array([h[i],v[j]])
            )
    data = np.array(data).astype('float32')
    data = vae.dec(data).as_ndarray()
    cv  = emb_cv(data)
    io.imsave('{}/map{}.png'.format(outdir, suffix), cv)

    # learning curve update
    shot_history = np.array(history)
    plt.clf()
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(shot_history[:,0], label='KL[train]')
    ax[0].plot(shot_history[:,-2], label='KL[test]')
    ax[0].set_ylim(bottom=0)
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(shot_history[:,1], label='ReconE[train]')
    ax[1].plot(shot_history[:,-1], label='ReconE[test]')
    ax[1].set_ylim(bottom=0)
    ax[1].legend()
    ax[1].grid()
    outname = '{}/curve.png'.format(outdir)
    plt.savefig(outname)