import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
class DenseNet(rm.Model):
    def __init__(self,
        input_shape,
        output_shape,
        units = 10,
        depth = 3,
        growth_rate = 12,
        dropout = False):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.depth = depth
        self.dropout = dropout
        parameters = []
        add_units = units
        for _ in range(depth-1):
            add_units += growth_rate
            parameters.append(rm.BatchNormalize())
            parameters.append(rm.Dense(add_units))
        self.hidden = rm.Sequential(parameters)
        self.input_batch = rm.BatchNormalize()
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
        hidden = self.input(rm.relu(self.input_batch(x)))
        for i in range(self.depth-1):
            sub = rm.relu(layers[i*2](hidden))
            sub = layers[i*2+1](sub)
            if self.dropout:
                sub = rm.dropout(sub)
            hidden = rm.concat(hidden, sub)
        if self.multi_output:
            layers = self.output._layers
            outputs = []
            for i in range(self.output_shape[0]):
                outputs.append(layers[i](hidden))
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

