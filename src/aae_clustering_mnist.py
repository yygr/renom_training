# encoding:utf-8
import gzip
from glob import glob
from pdb import set_trace
from time import time
from os import makedirs, path

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
target = 'aae'
model_id = 'semisupervised-clustering'
seed(10)
set_cuda_active(True) # gpu is mandatory
latent_dim = 2
epoch = 100
batch_size = 256
shot_freq = epoch//10
hidden = 1000
train = True
nb_superviser = 0#10000

# --- data loading & prepairing ---
data = np.load('mnist/data.npy')

y_train = data[0][0]
x_train = data[0][1].astype('float32')/255.
y_test = data[1][0]
x_test = data[1][1].astype('float32')/255.
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

def one_hot(data, size=10):
    temp = np.zeros((len(data), size))
    temp[np.arange(len(data)), data.reshape(-1)]=1
    return temp

y_train_1 = one_hot(y_train)
idx = random.permutation(len(y_train))[nb_superviser:]
y_train_1[idx,:] = 0

# --- model configuration ---

enc_base = rm.Sequential([
    #rm.BatchNormalize(),
    rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
    rm.BatchNormalize(),
    rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
    # xxx rm.BatchNormalize(),
    # Genの最後にBNを配置してはだめ
    rm.Dense(hidden), rm.Sigmoid()
    #rm.Dense(latent_dim, initializer=Uniform())
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
    mode='clustering', 
    prior='normal',
    label_dim=10,
    latent_dim = latent_dim, 
    hidden=200, 
)

dis_opt = rm.Adam(lr=0.001, b=0.5)
enc_opt = rm.Adam(lr=0.001, b=0.5)
cds_opt = rm.Adam(lr=0.001, b=0.9)

outdir='result/{}/{}'.format(target, model_id)
if not path.exists(outdir):
    makedirs(outdir)

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
        t0 = time()
        train_data = x_train[idx]
        with ae.train():
            ae(train_data, y=y_train_1[idx])
        t1 = time()
        with ae.enc.prevent_update():
            l = ae.gan_loss
            l.grad(detach_graph=False).update(dis_opt)
        t2 = time()
        with ae.enc.prevent_update():
            l = ae.CganE
            l.grad(detach_graph=False).update(cds_opt)
        t3 = time() 
        with ae.dis.prevent_update():
            # Force reconstruction error to small
            # for assigning latent location
            l = ae.enc_loss + ae.CgenE + .1*ae.reconE
            l.grad().update(enc_opt)
        t4 = time()
        batch_history.append([
            float(ae.reconE.as_ndarray()), 
            float(ae.gan_loss.as_ndarray()), 
            float(ae.enc_loss.as_ndarray()), 
            float(ae.real_count), 
            float(ae.fake_count), 
            float(ae.CganE.as_ndarray()),
            float(ae.CgenE.as_ndarray()),
            float(ae.Creal_count),
            float(ae.Cfake_count),
            t1-t0, t2-t1, t3-t2, t3-t0])
        mean = np.array(batch_history).mean(0)
        print_str = '>{:5d}/{:5d}'.format(offset, N)
        print_str += ' ReconE:{:.3f}'.format(mean[0])
        print_str += ' Dis:{:.3f}/{:.3f}={:.2f}/{:.2f}'.format(
            mean[1], mean[2], mean[3], mean[4]
        )
        print_str += ' Cds:{:.3f}/{:.3f}={:.2f}/{:.2f}'.format(
            mean[5], mean[6], mean[7], mean[8]
        )
        print_str += ' ETA:{:.1f}sec'.format((N-offset)/batch_size*mean[-1])
        print_str += ' {:.2f},{:.2f},{:.2f},{:.2f} {:>20}'.format(
            mean[9], mean[10], mean[11], mean[12], '')
        print(print_str, flush=True, end='\r')
    print_str = '#{:5d}/{:5d}'.format(e+1, epoch)
    print_str += ' ReconE:{:.3f}'.format(mean[0])
    print_str += ' Dis:{:.3f}/{:.3f}={:.2f}/{:.2f}'.format(
        mean[1], mean[2], mean[3], mean[4]
    )
    print_str += ' Cds:{:.3f}/{:.3f}={:.2f}/{:.2f}'.format(
        mean[5], mean[6], mean[7], mean[8]
    )
    print_str += ' @ {:.1f} sec {:>20}'.format(
        np.array(batch_history)[:,-1].sum(), ''
    )
    print(print_str)

    res, _ = ae.enc(x_test[:batch_size])
    res = res.as_ndarray()
    for i in range(batch_size, len(x_test), batch_size):
        z_mean, _ = ae.enc(x_test[i:i+batch_size])
        res = np.r_[res, z_mean.as_ndarray()]
    if latent_dim == 2 or e == epoch - 1:
        if latent_dim != 2:
            res = TSNE().fit_transform(res)
        plt.clf()
        plt.figure(figsize=(7,7))
        plt.scatter(res[:,0], res[:,1], c=y_test.reshape(-1), alpha=0.5)
        plt.axis('equal')
        plt.grid()
        if e % shot_freq == shot_freq - 1:
            outname = '{}/latent{:04d}.png'.format(outdir, e+1)
            plt.savefig(outname)
        outname = '{}/latent.png'.format(outdir)
        plt.savefig(outname)

    res_dim = 16
    ims = ae(x_test[:batch_size], 
        y=np.zeros((batch_size, 10))).as_ndarray()
    ims = ims.reshape(-1, 28, 28)
    og = x_test[:batch_size].reshape(-1, 28, 28)
    cv = np.zeros((res_dim*28, res_dim*28))
    for i in range(res_dim):
        for j in range(res_dim):
            idx = i*res_dim+j
            cv[i*28:(i+1)*28, j*28:(j+1)*28] = ims[idx] + og[idx]
    cv -= cv.min()
    cv /= cv.max()
    cv *= 255
    cv = cv.astype('uint8')
    if e % shot_freq == shot_freq - 1:
        outname = '{}/decode{}.png'.format(outdir, e+1)
        io.imsave(outname, cv)
    outname = '{}/decode.png'.format(outdir)
    io.imsave(outname, cv)