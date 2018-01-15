# encoding:utf-8
import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from time import time
from skimage import io
from glob import glob
import gzip
from renom.cuda.cuda import set_cuda_active
from numpy.random import seed, permutation
from sklearn.manifold import TSNE
from renom.utility.initializer import Uniform
from aae_func import AAE
from os import path, makedirs

model_id = 'AAE'

data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
#x_train = x_train * 2 - 1
x_test =  x_test.reshape(-1, 28*28)
#x_test = x_test * 2 - 1

# --- configuration ---
set_cuda_active(True)
seed(10)

latent_dim = 2
epoch = 100
batch_size = 256
model = 'AAE'
shot_freq = epoch // 10
hidden = 100
model_type = 'simple'
model_dist = 'uniform'
lr_rate = 1.
base_outdir = 'result/{}/{}/{}'.format(
    model_id, model_type, model_dist
)
if not path.exists(base_outdir):
    makedirs(base_outdir)

# --- model configuration ---
enc_base = rm.Sequential([
    rm.Dense(hidden), rm.Relu(), 
    rm.BatchNormalize(),
    rm.Dense(hidden), rm.Relu(),
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

ae = AAE(
    enc_base, dec, batch_size,
    latent_dim = latent_dim, 
    hidden=hidden, 
    prior=model_dist, 
    mode=model_type, 
)

dist_opt = rm.Adam(lr=0.001, b=0.5)
opt = rm.Adam(lr=0.001, b=0.5)

N = len(x_train)
batch_shape = (batch_size, 784)
history = []
for e in range(1, epoch+1):
    perm = permutation(N)
    batch_history = []
    for offset in range(0, N, batch_size):
        idx = perm[offset: offset+batch_size]
        start_time = time()
        train_data = np.zeros(batch_shape)
        tmp = x_train[perm[idx]]
        train_data[:len(tmp)] = tmp
        with ae.train():
            ae(train_data)
        with ae.enc.prevent_update():
            l = ae.gan_loss
            l.grad(detach_graph=False).update(dist_opt)
        with ae.dis.prevent_update():
            l = ae.enc_loss + lr_rate*ae.reconE
            l.grad().update(opt)
        batch_history.append([
            float(ae.gan_loss.as_ndarray()),
            float(ae.enc_loss.as_ndarray()),
            float(ae.reconE.as_ndarray()),
            float(ae.real_count),
            float(ae.fake_count),
            time() - start_time
        ])
        mean = np.array(batch_history).mean(0)
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
            mean[0], mean[1], mean[2]
        )
        print_str += ' {:.2f}/{:.2f}'.format(mean[3], mean[4])
        print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*mean[-1])
        print(print_str, flush=True, end='\r')
    print_str = '#{:5d}/{:5d}'.format(e, epoch)
    print_str += ' Dis:{:.3f} Enc:{:.3f} ReconE:{:.3f}'.format(
        mean[0], mean[1], mean[2]
    )
    print_str += ' {:.2f}/{:.2f}'.format(mean[3], mean[4])
    print_str += ' @ {:.1f}sec {:>20}'.format(
        np.array(batch_history)[:,-1].sum(), ''
    )
    print(print_str)
   

    # @@@ inference @@@
    ae.set_models(inference=True)
    res = ae.enc(x_test[:batch_size]).as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean = ae.enc(x_test[i:i+batch_size]).as_ndarray()
        res = np.r_[res, z_mean]
    ae.set_models(inference=False)
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.figure(figsize=(7,7))
        plt.scatter(res[:,0], res[:,1], 
            c=y_test.reshape(-1), alpha=0.5,
            label='epoch#{}'.format(e+1))
        plt.legend()
        plt.axis('equal')
        plt.grid()
        if e % shot_freq == shot_freq - 1:
            outname = '{}/test_latent{}.png'.format(base_outdir, e+1)
            plt.savefig(outname)
        outname = '{}/test_latent.png'.format(base_outdir)
        plt.savefig(outname)
        plt.close()

    ae.set_models(inference=True)
    res = ae.enc(x_train[:batch_size]).as_ndarray()
    for i in range(batch_size, len(x_train), batch_size):
        z_mean = ae.enc(x_train[i:i+batch_size]).as_ndarray()
        res = np.r_[res, z_mean]
    ae.set_models(inference=False)
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
        if e % shot_freq == shot_freq - 1:
            outname = '{}/train_latent{}.png'.format(base_outdir, e+1)
            plt.savefig(outname)
        outname = '{}/train_latent.png'.format(base_outdir)
        plt.savefig(outname)
        plt.close()


    res_dim = 16
    ae.set_models(inference=True)
    ims = ae(x_test[:batch_size]).as_ndarray()
    ae.set_models(inference=False)
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
        outname = '{}/test_decode{}.png'.format(base_outdir, e+1)
        io.imsave(outname, cv)
    outname = '{}/test_decode.png'.format(base_outdir)
    io.imsave(outname, cv)

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
            outname = '{}/test_map{}.png'.format(base_outdir, e+1)
            io.imsave(outname, cv)
        outname = '{}/test_map.png'.format(base_outdir)
        io.imsave(outname, cv)
