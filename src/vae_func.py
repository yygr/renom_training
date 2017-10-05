import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class keras_Enc(rm.Model):
    def __init__(self):
        self.hidden = rm.Sequential([
            rm.Conv2d(1, filter=2, padding=1),
            rm.Relu(),
            rm.Conv2d(64, filter=2, padding=1, stride=2),
            rm.Relu(),
            rm.Conv2d(64, filter=3, padding=1),
            rm.Relu(),
            rm.Conv2d(64, filter=3, padding=1),
            rm.Flatten(),
            rm.Dense(128),
            rm.Relu(),
        ])
        self.output = rm.Sequential([
            rm.Dense(2, initializer=Uniform()),
            rm.Dense(2, initializer=Gaussian(std=0.3)),
        ])
        self.latent_dim = 2
    def forward(self, x):
        x = self.hidden(x)
        #print("#", x.shape)
        z_mean = self.output._layers[0]
        z_log_var = self.output._layers[1]
        return z_mean(x), z_log_var(x)

class keras_Dec(rm.Model):
    def __init__(self):
        self.input = rm.Sequential([
           rm.Dense(64*14*14),
           rm.Relu(),
        ])
        self.hidden = rm.Sequential([
            rm.Deconv2d(64, filter=3, padding=1),
            rm.Relu(),
            rm.Deconv2d(64, filter=3, padding=1),
            rm.Relu(),
            rm.Deconv2d(64, filter=3, stride=2),
            rm.Relu(),
            rm.Conv2d(1, filter=2),
            rm.Sigmoid()
        ])
    def forward(self, x):
        hidden = self.input(x)
        #print(hidden.shape)
        hidden = rm.reshape(hidden, (len(x), 64, 14, 14))
        #print(hidden.shape)
        hidden = self.hidden(hidden)
        #print(hidden.shape)
        return hidden
       
class VGG_Enc(rm.Model):
    def __init__(
            self,
            input_shape = (28, 28),
            depth = 4,#28x28 -> 14x14 -> 7x7 -> 4x4 -> 2x2
            batch_normal = False,
            latent_dim = 10,
            dropout = False,
            intermidiate_dim = 128,
            max_channels = 64,
        ):
        self.depth = depth
        self.batch_normal = batch_normal
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.intermidiate_dim = intermidiate_dim
        self.max_channels = max_channels
        parameters = []
        channels = np.log(max_channels)/np.log(2)-self.depth + 1
        print('--- Ecoding Network ---')
        boot_steps = 0
        if channels <= 3:
            boot_steps = 3 - channels
            channels = 3
        channels = int(2**channels)
        boot_steps = int(boot_steps)
        dim = self.input_shape[0]
        for _ in range(boot_steps):
            if self.batch_normal:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
            else:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
            dim = dim // 2
            print('Max Pooling {}x{}'.format(dim, dim))
        for _ in range(self.depth - boot_steps):
            if self.batch_normal:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
            else:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
            channels *= 2
            dim = dim // 2
            print('Max Pooling {}x{}'.format(dim, dim))
        self.hidden = rm.Sequential(parameters)
        nb_parameters = dim * dim * channels
        print('Flatten {} params'.format(nb_parameters))
        parameters = []
        fcnn_depth = int((np.log(nb_parameters/intermidiate_dim))/np.log(4))
        nb_parameters = nb_parameters // 4 
        for _ in range(fcnn_depth):
            print('Dense {}u'.format(nb_parameters))
            parameters.append(rm.Dense(nb_parameters))
            nb_parameters = nb_parameters // 4 
        print('Dense {}u'.format(intermidiate_dim))
        parameters.append(rm.Dense(intermidiate_dim))
        print('*Mean Dense {}u'.format(latent_dim))
        parameters.append(rm.Dense(latent_dim, initializer=Uniform()))
        print('*Log Var Dense {}u'.format(latent_dim))
        parameters.append(rm.Dense(latent_dim, initializer=Gaussian(std=0.3)))
        self.fcnn = rm.Sequential(parameters)

    def forward(self, x):
        layers = self.hidden._layers
        for i in range(self.depth):
            if self.batch_normal:
                x = layers[i*4](x)
                x = rm.relu(layers[i*4+1](x))
                x = layers[i*4+2](x)
                x = rm.relu(layers[i*4+3](x))
            else:
                x = rm.relu(layers[i*2](x))
                #print(x.shape)
                x = rm.relu(layers[i*2+1](x))
                #print(x.shape)
            x = rm.max_pool2d(x, stride=2, padding=(1,1))
            #print(x.shape)
        x = rm.flatten(x)
        layers = self.fcnn._layers
        for i in range(len(layers[:-2])):
            x = rm.relu(layers[i](x))
            #print(x.shape)
            if self.dropout:
                x = rm.dropout(x, dropout_ratio=0.5)
        z_mean = layers[-2](x)
        z_log_var = layers[-1](x)
        return z_mean, z_log_var

class Dec(rm.Model):
    def __init__(
            self,
            latent_dim = 10,
            output_shape = (28, 28), # 正方行列を仮定
            #28x28 <- 14x14 <- 7x7 <-* 1024x2x2
            batch_normal = False,
            dropout = False,
            min_channels = 16,
        ):
        self.batch_normal = batch_normal
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.dropout = dropout
        self.min_channels = min_channels
        print('--- Decoding Network ---')
        parameters = []
        print_params = []
        dim = output_shape[0]
        channels = self.min_channels
        while dim%2 == 0 and dim > 2:
            parameters.append(rm.Deconv2d(
                channel=channels, stride=2, filter=2))
            if batch_normal:
                parameters.append(rm.BatchNormalize())
            dim = dim // 2
            print_params.append([dim, channels])
            channels *= 2
        if dim%2 == 1:
            parameters.append(rm.Deconv2d(
                channel=channels, stride=2, filter=3))
            dim = (dim - 1) // 2
            print_params.append([dim, channels])
            channels *= 2
        parameters.reverse()
        print_params.reverse()
        print('Dense {}x{}x{} & Reshape'.format(dim, dim,channels))
        self.channels = channels
        self.transform = rm.Dense(channels*1*dim*dim)
        for item in print_params:
            print('Deconv2d to {}x{} {}ch '.format(
                item[0], item[0], item[1]))
        self.hidden = rm.Sequential(parameters)
        self.output = rm.Conv2d(channel=1,stride=1,filter=1)
        print('Conv2d to {}x{} 1ch'.format(
            output_shape[0], output_shape[0]))
        self.dim = dim

    def forward(self, x):
        h = self.transform(x)
        #print(h.shape)
        h = rm.reshape(h, (len(x), self.channels, self.dim, self.dim))
        #print(h.shape)
        layers = self.hidden._layers
        for i in range(len(layers)):
            if self.batch_normal:
                h = layers[2*i](h)
                h = rm.relu(layers[2*i+1](h))
            else:
                h = rm.relu(layers[i](h))
            #print(h.shape)
        h = rm.sigmoid(self.output(h))
        return h

class Vae2d(rm.Model):
    def __init__(self, enc, dec):
        self.enc = enc
        self.dec = dec
        self.latent_dim = enc.latent_dim

    def forward(self, x):
        self.z_mean, self.z_log_var = self.enc(x)
        e = np.random.randn(len(x), self.latent_dim) * 1.
        z_new = self.z_mean + rm.exp(self.z_log_var/2)*e
        self.decoded = self.dec(z_new)
        nb, zd = self.z_log_var.shape
        self.kl_loss = - 0.5 * rm.sum(
            1 + self.z_log_var - self.z_mean**2 - rm.exp(self.z_log_var)
            )/nb
        #self.recon_loss = rm.sigmoid_cross_entropy(self.decoded, x) 
        self.recon_loss = rm.mean_squared_error(self.decoded, x) 
        vae_loss = self.kl_loss + self.recon_loss
        return vae_loss 

    
class DenseNet(rm.Model):
    def __init__(self,
        input_shape,
        output_shape,
        units = 10,
        depth = 3,
        growth_rate = 12,
        dropout = False,
        initializer=rm.utility.initializer.Gaussian(std=0.3),
        active=rm.tanh
        ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.depth = depth
        self.dropout = dropout
        self.active = active
        parameters = []
        add_units = units
        for _ in range(depth-1):
            add_units += growth_rate
            parameters.append(rm.BatchNormalize())
            parameters.append(rm.Dense(add_units,
                initializer=initializer
            ))
        self.hidden = rm.Sequential(parameters)
        self.input_batch = rm.BatchNormalize()
        self.input = rm.Dense(units)
        self.multi_output = False
        if isinstance(self.output_shape, tuple):
            self.multi_output = True
            parameters = []
            for _ in range(output_shape[0]):
                parameters.append(rm.BatchNormalize())
                parameters.append(rm.Dense(output_shape[1],
                    initializer=initializer
                ))
            self.output = rm.Sequential(parameters)
        else:
            self.output = rm.Dense(output_shape)
    
    def forward(self, x):
        layers = self.hidden._layers
        hidden = self.input(self.active(self.input_batch(x)))
        for i in range(self.depth-1):
            sub = self.active(layers[i*2](hidden))
            sub = layers[i*2+1](sub)
            if self.dropout:
                sub = rm.dropout(sub, dropout_ratio=0.2)
            hidden = rm.concat(hidden, sub)
        if self.multi_output:
            layers = self.output._layers
            outputs = []
            for i in range(self.output_shape[0]):
                hidden=self.active(layers[i*2](hidden))
                outputs.append(layers[i*2+1](hidden))
            return outputs
        return self.output(hidden)


class EncoderDecoder(rm.Model):
    def __init__(self,
        input_shape,
        output_shape,
        units = 10,
        depth = 3,
        batch_normal = False,
        dropout = False):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.depth = depth
        self.batch_normal = batch_normal
        self.dropout = dropout
        parameters = []
        for _ in range(depth-1):
            if self.batch_normal:
                parameters.append(rm.BatchNormalize())
            parameters.append(rm.Dense(units))
        self.hidden = rm.Sequential(parameters)
        self.input = rm.Dense(units)
        self.multi_output = False
        if isinstance(self.output_shape, tuple):
            self.multi_output = True
            parameters = []
            for _ in range(output_shape[0]):
                parameters.append(rm.Dense(output_shape[1]))
            self.output = rm.Sequential(parameters)
        else:
            self.output = rm.Dense(output_shape)
    
    def forward(self, x):
        layers = self.hidden._layers
        hidden = self.input(x)
        for i in range(self.depth-1):
            if self.batch_normal:
                hidden = layers[i*2](hidden)
                hidden = rm.sigmoid(layers[i*2+1](hidden))
            else:
                hidden = rm.sigmoid(layers[i](hidden))
            if self.dropout:
                hidden = rm.dropout(hidden)
        if self.multi_output:
            layers = self.output._layers
            outputs = []
            for i in range(self.output_shape[0]):
                outputs.append(layers[i](hidden))
            return outputs
        return self.output(hidden)

class Vae(rm.Model):
    def __init__(self, enc, dec, output_shape = 1):
        self.enc = enc
        self.dec = dec
        self.latent_dimension = enc.output_shape[-1]

    def forward(self, x):
        z = self.enc(x)
        self.z_mean, self.z_log_var = z[0], z[1]#rm.relu(z[1])
        e = np.random.randn(len(x), self.latent_dimension) * 1.
        z_new = self.z_mean + rm.exp(self.z_log_var/2)*e
        d = self.dec(z_new)
        self.d_mean, self.d_log_var = d[0], d[1]#rm.relu(d[1])
        nb, zd = self.z_log_var.shape
        if 0:
            kl_loss = rm.Variable(0)
            pxz_loss = rm.Variable(0)
            for i in range(nb):
                kl_loss += -0.5*rm.sum(
                    1 + self.z_log_var[i] - self.z_mean[i]**2 - rm.exp(self.z_log_var[i])
                    )
                pxz_loss += rm.sum(
                    0.5*self.d_log_var[i] + (x[i]-self.d_mean[i])**2/(2*rm.exp(self.d_log_var[i]))
                    )
            #pxz_loss = rm.sum(0.5*d_log_var + (x-d_mean)**2/(2*rm.exp(d_log_var)))
            vae_loss = (kl_loss + pxz_loss)/nb
        else:
            kl_loss = - 0.5 * rm.sum(
                1 + self.z_log_var - self.z_mean**2 - rm.exp(self.z_log_var)
                )
            pxz_loss = rm.sum(
                0.5 * self.d_log_var + (x-self.d_mean)**2/(2*rm.exp(self.d_log_var))
                ) 
            vae_loss = (kl_loss + pxz_loss)/nb
        return vae_loss 

