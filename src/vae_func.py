import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class VGG(rm.Model):
    def __init__(
            self,
            nname='',
            batch_size = 64,
            input_shape = (1, 28, 28),
            batchnormal = True,
            dropout = False,
            first_channel = 8,
            growth_factor = 2,
            repeats = 2,
            tgt_dim = 4,
            tgt_parameters = 3000,
            keep_vertical = False,
            check_network = False,
            act = rm.Relu(),
        ):
        self.input_shape = input_shape
        self.keep_v = keep_vertical
        self.dropout = dropout
        self.batchnormal = batchnormal
        self.act = act
        if check_network:
            print('--- {} Network ---'.format(nname))
        ch = first_channel
        in_ch, v_dim, h_dim = self.input_shape
        if check_network:
            print('Input {}'.format(input_shape))
        def check_continue():
            v_deci = False if keep_vertical else v_dim > tgt_dim
            h_deci = h_dim > tgt_dim
            c_deci = ch * v_dim * h_dim > tgt_parameters
            return c_deci and v_deci or h_deci
        parameters = []
        repeat_cnt = 0 
        self.first = None
        while check_continue():
            if self.first and batchnormal:
                parameters.append(rm.BatchNormalize())
                if check_network:
                    print('BN ', end='')
            cnn_layer = rm.Conv2d(
                    channel=ch,
                    filter=3,
                    padding=1,
            )
            if self.first:
                parameters.append(cnn_layer)
            else:
                self.first = cnn_layer
            if check_network:
                print('Conv2d -> {}'.format((
                    batch_size, ch, v_dim, h_dim)))
            repeat_cnt += 1
            if repeat_cnt == repeats:
                repeat_cnt = 0
                ch = int(ch*growth_factor)
                stride = (1 if self.keep_v else 2, 2)
                parameters.append(
                    rm.Conv2d(
                        channel=ch,
                        filter=1,
                        stride=stride,
                ))
                v_dim = v_dim if self.keep_v else int(np.ceil(v_dim/2))
                h_dim = int(np.ceil(h_dim/2))
                if check_network:
                    print('Conv2d -> {}'.format((
                        batch_size, ch, v_dim, h_dim)))
        self.output_shape = (batch_size, ch, v_dim, h_dim)
        self.parameters = rm.Sequential(parameters)
        self.nb_parameters = ch*v_dim*h_dim
        if check_network:
            self.forward(
                np.zeros((batch_size,
                input_shape[0], input_shape[1], input_shape[2])),
                check_network=True,
            )
    def forward(self, x, check_network=False):
        hidden = x
        if check_network:
            print('^ {}'.format(hidden.shape))
        hidden = self.first(hidden)
        if check_network:
            print('^ {}'.format(hidden.shape))
        layers = self.parameters
        for i, layer in enumerate(layers):
            if check_network:
                print('^ {}'.format(hidden.shape))
            hidden = layer(hidden)
            if self.batchnormal:
                if i % 2 == 1:
                    hidden = self.act(hidden)
            else:
                hidden = self.act(hidden)
        if check_network:
            print('^ {}'.format(hidden.shape))
        return rm.flatten(hidden)

class Dec2d(rm.Model):
    def __init__(
            self,
            input_params = 32768,
            first_shape = (1, 32, 32, 32),
            output_shape = (1, 1, 64, 64),
            check_network = False,
            batchnormal = True,
            dropout = False,
            down_factor = 1.6,
            act = rm.Relu(),
            last_act = rm.Sigmoid(),
        ):
        self.input_params = input_params
        self.latent_dim = input_params
        self.first_shape = first_shape
        self.output_shape = output_shape
        self.act = act
        self.last_act = last_act
        self.down_factor = down_factor
        def decide_factor(src, dst):
            factor = np.log(src/dst)/np.log(2)
            if factor%1 == 0:
                return factor
            return np.ceil(factor)
        ch = first_shape[1]
        v_factor = decide_factor(output_shape[2], first_shape[2])
        h_factor = decide_factor(output_shape[3], first_shape[3])
        v_dim, h_dim = first_shape[2], first_shape[3]
        parameters = []
        check_params = np.array(first_shape[1:]).prod()
        self.trans = False 
        if input_params != check_params:
            if check_network:
                print('--- Decoder Network ---')
                print('inserting Dense({})'.format(check_params))
            self.trans = rm.Dense(check_params)
        while v_factor != 0 or h_factor != 0:
            if batchnormal:
                parameters.append(rm.BatchNormalize())
                if check_network:
                    print('BN ', end='')
            stride = (2 if v_factor>0 else 1, 2 if h_factor>0 else 1)
            if check_network:
                print('transpose2d ch={}, filter=2, stride={}'.format(
                    ch, stride))
            parameters.append(rm.Deconv2d(
                channel=ch,
                filter=2, stride=stride))
            if self.act:
                parameters.append(self.act)
            if ch > output_shape[1]:
                ch = int(np.ceil(ch / self.down_factor))
            v_dim = v_dim*2 if v_factor>0 else v_dim+1
            h_dim = h_dim*2 if h_factor>0 else h_dim+1
            v_factor = v_factor-1 if v_factor>0 else 0
            h_factor = h_factor-1 if h_factor>0 else 0
        if v_dim>output_shape[2] or h_dim>output_shape[2]:
            last_filter = (
                    v_dim-output_shape[2]+1,
                    h_dim-output_shape[3]+1)
            if check_network:
                print('conv2d filter={}, stride=1'.format(last_filter))
            parameters.append(rm.Conv2d(
                channel=output_shape[1],
                filter=last_filter,stride=1))
        self.parameters = rm.Sequential(parameters)
        if check_network:
            self.forward(
                np.zeros((first_shape[0],input_params)),
                print_parameter=True
            )

    def forward(self, x, print_parameter=False):
        hidden = x
        if print_parameter:
            print('^ {}'.format(hidden.shape))
        if self.trans:
            hidden = self.trans(hidden)
            if print_parameter:
                print('^ {}'.format(hidden.shape))
        hidden = rm.reshape(hidden, self.first_shape)
        if print_parameter:
            print('^ {}'.format(hidden.shape))
        layers = self.parameters
        for layer in layers:
            hidden = layer(hidden)
            if print_parameter:
                print('^ {}'.format(hidden.shape))
        if self.act:
            hidden = self.last_act(hidden)
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
        boot_th = np.log(self.input_shape[0])/np.log(2)
        print('--- Ecoding Network ---')
        boot_steps = 0
        if boot_th < self.depth:
            boot_steps = int(self.depth - boot_th) + 1
            channels = 3 
        channels = int(2**channels)
        boot_steps = int(boot_steps)
        dim = self.input_shape[0]
        for _ in range(boot_steps):
            if self.batch_normal:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                print('Batch Normalize')
                parameters.append(rm.BatchNormalize())
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                print('Batch Normalize')
                parameters.append(rm.BatchNormalize())
            else:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
        for i in range(self.depth - boot_steps):
            if self.batch_normal:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                print('Batch Normalize')
                parameters.append(rm.BatchNormalize())
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append(rm.Conv2d(channels, padding=(1,1)))
                print('Batch Normalize')
                parameters.append(rm.BatchNormalize())
            else:
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
                print('Conv2d {}x{} {}ch'.format(dim, dim, channels))
                parameters.append((rm.Conv2d(channels, padding=(1,1))))
            channels *= 2
            dim = dim // 2
            if i == self.depth - boot_steps - 1:
                print('Average Pooling {}x{}'.format(dim, dim))
            else:
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
            if i == self.depth - 1:
                x = rm.average_pool2d(x, stride=2, padding=(1,1))
            else:
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

class Densenet_Enc(rm.Model):
    def __init__(
            self,
            input_shape = (28, 28),
            blocks = 2,
            depth = 3,
            growth_rate = 12,
            latent_dim = 10,
            dropout = False,
            intermidiate_dim = 128,
            compression = 0.5,
            initial_channel = 8
        ):
        self.depth = depth
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.intermidiate_dim = intermidiate_dim
        self.compression = compression
        self.growth_rate = growth_rate
        self.blocks = blocks
        print('--- Ecoding Network ---')
        parameters = []
        channels = initial_channel
        dim = self.input_shape[0]
        print('Input image {}x{}'.format(dim, dim))
        dim = dim // 2
        print(' Conv2d > {}x{} {}ch'.format(dim,dim,channels))
        self.input = rm.Conv2d(channels, filter=5, padding=2, stride=2)
        for _ in range(blocks):
            t_params, channels = self.denseblock(
                    dim=dim,
                    input_channels=channels,
            )
            parameters += t_params
            dim = (dim+1) // 2
        self.hidden = rm.Sequential(parameters)
        nb_parameters = dim * dim * channels
        print(' Flatten {} params'.format(nb_parameters))
        parameters = []
        fcnn_depth = int((np.log(nb_parameters/intermidiate_dim))/np.log(4))
        nb_parameters = nb_parameters // 4 
        for _ in range(fcnn_depth):
            print(' Dense {}u'.format(nb_parameters))
            parameters.append(rm.Dense(nb_parameters))
            nb_parameters = nb_parameters // 4 
        print(' Dense {}u'.format(intermidiate_dim))
        parameters.append(rm.Dense(intermidiate_dim))
        print('*Mean Dense {}u'.format(latent_dim))
        parameters.append(rm.Dense(latent_dim, initializer=Uniform()))
        print('*Log Var Dense {}u'.format(latent_dim))
        parameters.append(rm.Dense(latent_dim, initializer=Gaussian(std=0.3)))
        self.fcnn = rm.Sequential(parameters)

    def denseblock(self,
        dim = 8,
        input_channels=10,
        dropout=False
        ):
        parameters = []
        c = input_channels
        print('-> {}'.format(c))
        for _ in range(self.depth):
            c += self.growth_rate
            print('Batch Normalize')
            parameters.append(rm.BatchNormalize())
            print(' Conv2d > {}x{} {}ch'.format(
                dim, dim, self.growth_rate 
            ))
            parameters.append(rm.Conv2d(
                self.growth_rate, filter=3, padding=(1,1)
            ))
            if self.dropout:
                print('Dropout')
        c = int(c*self.compression)
        print('*Conv2d > {}x{} {}ch'.format(
            dim, dim, c 
        ))
        parameters.append(rm.Conv2d(
            c, filter=1
        ))
        print(' Average Pooling')
        print('<- {}'.format(c))
        return parameters, c 


    def forward(self, x):
        hidden = self.input(x)
        #print(hidden.shape)
        hidden = rm.max_pool2d(hidden, stride=1, padding=1)
        #print(hidden.shape)
        layers = self.hidden._layers
        for i in range(self.blocks):
            offset = i*(self.depth*2+1)
            for j in range(self.depth):
                sub = rm.relu(layers[offset+2*j](hidden))
                #print('{}.{} b {}'.format(i,j,sub.shape))
                sub = layers[offset+2*j+1](sub)
                #print('{}.{} + {}'.format(i,j,sub.shape))
                if self.dropout:
                    sub = rm.dropout(sub)
                hidden = rm.concat(hidden, sub)
                #print('{}.{} = {}'.format(i,j,hidden.shape))
            offset = (i+1)*(self.depth*2+1)-1
            hidden = layers[offset](hidden)
            #print('{}.{} - {}'.format(i,j,hidden.shape))
            hidden = rm.average_pool2d(hidden, stride=2, padding=1)
            #print('{}.{} > {}'.format(i,j,hidden.shape))
        x = rm.flatten(hidden)
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


class Enc(rm.Model):
    def __init__(
        self, pre, latent_dim,
        output_act = None,
        ):
        self.pre = pre
        self.latent_dim = latent_dim
        self.zm_ = rm.Dense(latent_dim)
        self.zlv_ = rm.Dense(latent_dim)
        self.output_act = output_act
    def forward(self, x):
        hidden = self.pre(x)
        self.zm = self.zm_(hidden)
        self.zlv = self.zlv_(hidden)
        if self.output_act:
            self.zm = self.output_act(self.zm) 
            self.zlv = self.output_act(self.zlv)
        return self.zm

class VAE(rm.Model):
    def __init__(
            self, 
            enc,
            dec,
            latent_dim, 
        ):
        self.latent_dim = latent_dim
        self.enc = enc
        self.dec = dec

    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.enc(x)
        e = np.random.randn(nb, self.latent_dim)
        self.z = self.enc.zm + rm.exp(self.enc.zlv/2)*e
        self.decd = self.dec(self.z)
        self.reconE = rm.mean_squared_error(self.decd, x)
        self.kl_loss = - 0.5 * rm.sum(
            1 + self.enc.zlv - self.enc.zm**2 -rm.exp(self.enc.zlv)
        )/nb
        return self.decd

