import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
np.set_printoptions(precision=3)

random.seed(10)

N = 30
noise_rate = 0.3
x_noise_rate = 0.1
epoch = 1000
batch_size = 5#'Full'
vae_learn = True

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

original_dimension = 2
latent_dimension = 1 

def sampling(args):
    z_mean, z_log_var = args
    e = K.random_normal(shape=(K.shape(z_mean)[0], 1, latent_dimension), 
        mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var/2) * e

_x = Input(shape=(original_dimension,))
#x = gen_network(_x)
x = Dense(128)(_x)
z_mean = Dense(latent_dimension, activation='sigmoid')(x)
z_log_var = Dense(latent_dimension, activation='relu')(x)
z = Lambda(sampling, output_shape=(latent_dimension,))([
    z_mean, z_log_var#, latent_dimension
])
#x = gen_network(z)
x = Dense(128)(_x)
dec_mean = Dense(original_dimension, activation='sigmoid')(x)
dec_log_var = Dense(original_dimension, activation='relu')(x)

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

y = VaeLoss()([_x, z_mean, z_log_var, dec_mean, dec_log_var])
vae = Model(_x, y)
vae.compile(optimizer='sgd', loss=None)

vae.fit(x_train, shuffle=True, epochs=10000, 
batch_size=batch_size, validation_data=(x_test, None))

set_trace()

enc = EncoderDecoder(2, (2, latent_dimension), units=4, depth=2)
dec = EncoderDecoder(latent_dimension, (2, 2), units=4, depth=2)
vae = Vae(enc, dec)

optimizer = rm.Adam()#Sgd(lr=0.3, momentum=0.4)
plt.clf()
epoch_splits = 10
epoch_period = epoch // epoch_splits
fig, ax = plt.subplots(epoch_splits, 2, 
figsize=(16, epoch_splits*8))
if batch_size == 'Full':
    batch_size = len(train_x)

curve = []
neighbor_period = []
for e in range(epoch):
    s = time()
    perm = np.random.permutation(len(train_x))
    batch_loss = []
    for i in range(0, len(train_x), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = train_x[idx]
        batch_y = train_y[idx]
        with vae.train():
            vae_loss = vae(np.c_[batch_x, batch_y])
        grad = vae_loss.grad()
        grad.update(optimizer)
        batch_loss.append(vae_loss.as_ndarray())
    neighbor_period.append(time()-s)
    test_loss = vae(np.c_[test_x, test_y])
    curve.append([np.array(batch_loss).mean(),test_loss.as_ndarray()])
    if e % epoch_period == epoch_period - 1 or e == epoch:
        if e > epoch//5:
            vae_learn = False
        current_period =  np.array(neighbor_period).mean()
        neighbor_period = []
        ax_ = ax[e//epoch_period]
        curve_na = np.array(curve)
        ax_[0].text(0,0.5, '{:.2f}sec @ epoch'.format(current_period))
        ax_[0].plot(curve_na[:,0])
        ax_[0].plot(curve_na[:,1])
        ax_[0].set_ylim(0,3)
        ax_[0].set_xlim(-2, epoch+10)
        ax_[0].grid()
        vae(np.c_[train_x, train_y])
        pred_train = vae.d[0]
        vae(np.c_[test_x, test_y])
        pred_test = vae.d[0]
        ax_[1].plot(x_axis_org, base, 'k-')
        ax_[1].scatter(x_axis, y_axis, marker='+')
        ax_[1].scatter(pred_train[:,0], pred_train[:,1], c='g', alpha=0.3)
        ax_[1].scatter(pred_test[:,0], pred_test[:,1], c='r', alpha=0.6)
        ax_[1].grid()
        plt.pause(0.5)
fig.savefig('result/vae{}.png'.format(epoch))
plt.pause(3)
