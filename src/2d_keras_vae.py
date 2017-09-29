import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from keras.layers import Input, Dense, Lambda, Layer
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import backend as K
np.set_printoptions(precision=3)

random.seed(10)

N = 100
noise_rate = 0.3
x_noise_rate = 0.1
epoch = 5000
batch_size = 10#'Full'
vae_learn = True
original_dimension = 2
latent_dimension = 1
intermidiate_dimension = 6

if 0:
    z_true = np.linspace(0,1,N)
    noise = np.random.randn(N)
    noise = np.abs(noise)
    noise /= noise.max()
    z_noised = z_true + noise
    r = np.power(z_noised,0.5)
    phi = 0.25 * np.pi * z_noised
    x1 = r*np.cos(phi) - 0.5
    x2 = r*np.sin(phi) - 0.5

    r_true = np.power(z_true, 0.5)
    phi_true = 0.25 * np.pi * (z_true)
    x1_true = r_true * np.cos(phi_true) - 0.5
    x2_true = r_true * np.sin(phi_true) - 0.5
    Y = np.transpose(np.reshape((x1_true,x2_true), (2,N)))

    x1 = np.random.normal(x1, 0.1*np.power(z_noised,2), N)
    x2 = np.random.normal(x2, 0.1*np.power(z_noised,2), N)
else:
    z_true = np.random.uniform(0,1,N)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r*np.cos(phi)
    x2 = r*np.sin(phi)

    x1 = np.random.normal(x1, 0.10*np.power(z_true,2), N)
    x2 = np.random.normal(x2, 0.10*np.power(z_true,2), N)
X = np.transpose(np.reshape((x1,x2), (2, N)))
X = np.asarray(X, dtype='float32')
if 0:
    plt.scatter(X[:,0],X[:,1], marker='.', alpha=0.2)
    #plt.plot(Y[:,0], Y[:,1], alpha=0.2, c='r')
    plt.show()

idx = random.permutation(N)
x_train = X[idx[::2]] 
x_test = X[idx[1::2]]

def gen_network(hidden, units=4, depth=2):
    for _ in range(depth):
        hidden = Dense(units, activation='relu')(hidden)
    return hidden


def sampling(args):
    z_mean, z_log_var = args
    e = K.random_normal(shape=(K.shape(z_mean)[0], latent_dimension), 
        mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var/2) * e

_x = Input(shape=(original_dimension,))
#x = gen_network(_x)
hidden = Dense(intermidiate_dimension, activation='sigmoid')(_x)
hidden = Dense(intermidiate_dimension, activation='sigmoid')(hidden)#, activation='sigmoid')(hidden)
z_mean = Dense(latent_dimension)(hidden)
z_log_var = Dense(latent_dimension)(hidden)#, activation='relu')(hidden)
z = Lambda(sampling, output_shape=(latent_dimension,))([
    z_mean, z_log_var#, latent_dimension
])
#x = gen_network(z)
x = Dense(intermidiate_dimension, activation='sigmoid')(z)
x = Dense(intermidiate_dimension, activation='sigmoid')(x)#, activation='sigmoid')(x)
dec_mean = Dense(original_dimension)(x)
dec_log_var = Dense(original_dimension)(x)#, activation='relu')(x)

class VaeLoss(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VaeLoss, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, z_m, z_l_v, d_m, d_l_v = inputs
        kl_loss =  - 0.5 * K.sum(1. + z_l_v - K.square(z_m) - K.exp(z_l_v), axis=1)
        dec_loss = K.sum(0.5 * d_l_v + 0.5 * K.square(x - d_m) / K.exp(d_l_v), axis=1)
        vae_loss = K.mean(kl_loss + dec_loss)
        self.add_loss(vae_loss, inputs=inputs)
        return vae_loss

con = Concatenate(axis=-1)

y = VaeLoss()([_x, z_mean, z_log_var, dec_mean, dec_log_var])
vae = Model(_x, y)
vae.compile(optimizer='adam', loss=None)
enc = Model(_x, con([z_mean,z_log_var]))
vae_ = Model(_x, con([dec_mean,dec_log_var]))
#vae.summary()

plt.clf()
epoch_splits = 5
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(16, epoch_splits*4))
if batch_size == 'Full':
    batch_size = len(train_x)

loss_curve = []
val_loss_curve = []
neighbor_period = []
for e in range(epoch_splits):
    s = time()
    hist = vae.fit(x_train, shuffle=True, epochs=epoch_period, 
        batch_size=batch_size, validation_data=(x_test, None))
    loss_curve += hist.history['loss']
    val_loss_curve += hist.history['val_loss']
    ax_ = ax[e]
    ax_[0].text(0,0.5, '{:.3f}sec @ epoch'.format((time()-s)/epoch_period))
    ax_[0].plot(loss_curve)
    ax_[0].plot(val_loss_curve)
    ax_[0].set_ylim(-4,1)
    ax_[0].set_xlim(-2, epoch+10)
    ax_[0].grid()
    pred_train = vae_.predict(x_train)
    pred_test = vae_.predict(x_test)
    #z = enc.predict(x_train[:batch_size])
    #d = vae_.predict(x_train[:batch_size])
    #print(vae.predict(x_train[:batch_size]))
    #l1 = - 0.5 * np.sum(1. + z[:,1:] - z[:,:1]**2 - np.exp(z[:,1:]), 1)
    #l2 = np.sum(0.5 * d[:,2:] + 0.5 * (x_train[:batch_size] - d[:,:2])**2 / np.exp(d[:,2:]), 1)
    #print(l1)
    #print(l2)
    #print(l1+l2)
    #set_trace()
    #ax_[1].plot(Y[:,0], Y[:,1], 'k-')
    ax_[1].scatter(x_train[:,0], x_train[:,1], marker='x', alpha=0.3, c='b')
    ax_[1].scatter(pred_train[:,0], pred_train[:,1], c='g', alpha=0.2, marker='.')
    ax_[1].scatter(pred_test[:,0], pred_test[:,1], c='r', alpha=0.2, marker='.')
    """
    ax_[1].errorbar(pred_train[:,0], pred_train[:,1], 
        xerr=pred_train[:,2], yerr=pred_train[:,3], c='g', alpha=0.6, fmt='.')
    ax_[1].errorbar(pred_test[:,0], pred_test[:,1], 
        xerr=pred_test[:,2], yerr=pred_train[:,3], c='r', alpha=0.6, fmt='.')
    """
    ax_[1].grid()
    #set_trace()
    plt.pause(0.5)
fig.savefig('result/2d_keras{}.png'.format(epoch))
plt.pause(3)

weights = vae.get_weights()
for item in weights:
    print('{}'.format(item))
