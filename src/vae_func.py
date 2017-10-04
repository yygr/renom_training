import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time

class VGG_Enc(rm.Model):
    def __init__(
            self,
            input_shape = (28, 28),
            depth = 4,#28x28 -> 14x14 -> 7x7 -> 4x4 -> 2x2
            batch_normal = False,
            latent_dim = 10,
            dropout = False
        ):
        self.depth = depth
        self.batch_normal = batch_normal
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout = dropout
        parameters = []
        channels = np.log(1024)/np.log(2)-self.depth + 1
        boot_steps = 0
        if channels <= 3:
            boot_steps = 3 - channels
            channels = 3
        channels = int(2**channels)
        for _ in range(boot_steps):
            if self.batch_normal:
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
            else:
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
        for _ in range(self.depth - boot_steps):
            if self.batch_normal:
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                parameters.append(rm.BatchNormalize())
            else:
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
            channels *= 2
        self.hidden = rm.Sequential(parameters)
        nb_parameters = int(
                np.log(input_shape[0]*input_shape[1])
                / np.log(self.depth)
            )*1024
        parameters = []
        fcnn_depth = int((np.log(nb_parameters) - np.log(latent_dim))/np.log(4))
        nb_parameters = nb_parameters // 4 
        for _ in range(fcnn_depth):
            parameters.append(rm.Dense(nb_parameters))
            nb_parameters = nb_parameters // 4 
        parameters.append(rm.Dense(latent_dim))
        parameters.append(rm.Dense(latent_dim))
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
            x = rm.sigmoid(layers[i](x))
            #print(x.shape)
            if self.dropout:
                x = rm.dropout(x, dropout_ratio=0.5)
        z_mean = rm.tanh(layers[-2](x))
        z_log_var = rm.tanh(layers[-1](x))
        return z_mean, z_log_var

class Dec(rm.Model):
    def __init__(
            self,
            latent_dim = 10,
            output_shape = (28, 28), # 正方行列を仮定
            #28x28 <- 14x14 <- 7x7 <-* 1024x2x2
            batch_normal = False,
            dropout = False
        ):
        self.batch_normal = batch_normal
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.dropout = dropout
        parameters = []
        dim = output_shape[0]
        channels = 16 
        while dim%2 == 0 and dim > 2:
            parameters.append(rm.Deconv2d(
                channel=channels, stride=2, filter=2))
            if batch_normal:
                parameters.append(rm.BatchNormalize())
            dim = dim // 2
            channels *= 4
        if dim%2 == 1:
            parameters.append(rm.Deconv2d(
                channel=channels, stride=2, filter=3))
            dim = (dim - 1) // 2
            channels *= 4
        parameters.reverse()
        self.hidden = rm.Sequential(parameters)
        self.transform = rm.Dense(1024*1*dim*dim)
        self.output = rm.Conv2d(channel=1,stride=1,filter=1)
        self.dim = dim

    def forward(self, x):
        h = self.transform(x)
        #print(h.shape)
        h = rm.reshape(h, (len(x), 1024, self.dim, self.dim))
        #print(h.shape)
        layers = self.hidden._layers
        for i in range(len(layers)):
            if self.batch_normal:
                h = layers[2*i](h)
                h = rm.relu(layers[2*i+1](h))
            else:
                h = rm.relu(layers[i](h))
            #print(h.shape)
        #h = rm.sigmoid(self.output(h))
        return self.output(h)

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
            )
        self.recon_loss = rm.sigmoid_cross_entropy(self.decoded, x) 
        vae_loss = self.kl_loss/nb + self.recon_loss
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

