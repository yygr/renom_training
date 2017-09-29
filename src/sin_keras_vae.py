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
epoch = 2000
batch_size = 10#'Full'
vae_learn = True
original_dimension = 2
latent_dimension = 1
intermidiate_dimension = 10

noise = random.randn(N).astype('float32')*noise_rate
x_axis_org = np.linspace(-np.pi,np.pi,N).astype('float32')
base = np.sin(x_axis_org).astype('float32')
x_axis = x_axis_org + random.randn(N).astype('float32')*x_noise_rate
y_axis = base+noise
x_axis = x_axis.reshape(N, 1)
y_axis = y_axis.reshape(N, 1)
idx = random.permutation(N)
train_idx = idx[::2]
test_idx = idx[1::2]
train_x = x_axis[train_idx]
train_y = y_axis[train_idx]
test_x = x_axis[test_idx]
test_y = y_axis[test_idx]
x_train = np.c_[train_x, train_y]
x_test = np.c_[test_x, test_y]

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
hidden = gen_network(_x, units=100, depth=2)
#hidden = Dense(intermidiate_dimension, activation='sigmoid')(_x)
#hidden = Dense(intermidiate_dimension, activation='sigmoid')(hidden)
z_mean = Dense(latent_dimension)(hidden)
z_log_var = Dense(latent_dimension)(hidden)
z = Lambda(sampling, output_shape=(latent_dimension,))([
    z_mean, z_log_var#, latent_dimension
])
x = gen_network(z, units=100, depth=2)
#x = Dense(intermidiate_dimension, activation='sigmoid')(z)
#x = Dense(intermidiate_dimension, activation='sigmoid')(x)
dec_mean = Dense(original_dimension)(x)
dec_log_var = Dense(original_dimension)(x)

class VaeLoss(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VaeLoss, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, z_m, z_l_v, d_m, d_l_v = inputs
        kl_loss = - 0.5 * K.sum(1. + z_l_v - K.square(z_m) - K.exp(z_l_v), axis=1)
        dec_loss = K.sum(0.5 * d_l_v + 0.5 * K.square(x - d_m) / K.exp(d_l_v), axis=1)
        vae_loss = K.mean(kl_loss + dec_loss)
        self.add_loss(vae_loss, inputs=inputs)
        return vae_loss

con = Concatenate(axis=-1)

y = VaeLoss()([_x, z_mean, z_log_var, dec_mean, dec_log_var])
vae = Model(_x, y)
vae.compile(optimizer='adam', loss=None)
enc = Model(_x, z_mean)
vae_ = Model(_x, con([dec_mean,dec_log_var]))
#vae.summary()

plt.clf()
epoch_splits = 5
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(16, epoch_splits*8))
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
    ax_[0].set_ylim(-3,3)
    ax_[0].set_xlim(-2, epoch+10)
    ax_[0].grid()
    pred_train = vae_.predict(x_train)
    pred_test = vae_.predict(x_test)
    ax_[1].plot(x_axis_org, base, 'k-')
    xerr = np.exp(pred_train[:,2]/2)
    yerr = np.exp(pred_train[:,3]/2)
    ax_[1].errorbar(pred_train[:,0], pred_train[:,1], 
        xerr=xerr, yerr=yerr, 
        ecolor='skyblue', c='g', alpha=0.6, fmt='.')
    xerr = np.exp(pred_test[:,2]/2)
    yerr = np.exp(pred_test[:,3]/2)
    ax_[1].errorbar(pred_test[:,0], pred_test[:,1], 
        xerr=xerr, yerr=yerr, 
        ecolor='skyblue', c='r', alpha=0.6, fmt='.')
    ax_[1].scatter(x_axis, y_axis, marker='x', alpha=0.3, c='b')
    ax_[1].grid()
    plt.pause(0.5)
fig.savefig('result/keras{}.png'.format(epoch))
plt.pause(3)

weights = vae.get_weights()
for item in weights:
    print('{}'.format(item))
