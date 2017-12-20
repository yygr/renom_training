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
        self.first = None
        def check_continue():
            v_deci = False if keep_vertical else v_dim > tgt_dim
            h_deci = h_dim > tgt_dim
            return v_deci or h_deci
        parameters = []
        repeat_cnt = 0 
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



class Densenet(rm.Model):
    def __init__(
            self,
            nname='',
            input_shape = (1, 28, 28),
            blocks = 2,
            depth = 3,
            growth_rate = 12,
            latent_dim = 10,
            dropout = False,
            intermidiate_dim = 128,
            first_filter = 5,
            compression = 0.5,
            initial_channel = 8,
            keep_vertical = False,
            check_network = False,
            batch_size = 64,
            print_network = True,
            # pooling type
        ):
        self.depth = depth
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.intermidiate_dim = intermidiate_dim
        self.compression = compression
        self.growth_rate = growth_rate
        self.blocks = blocks
        self.keep_v = keep_vertical
        if print_network:
            print('--- {} Network ---'.format(nname))
        parameters = []
        channels = initial_channel
        in_ch, dim_v, dim_h = self.input_shape
        if print_network:
            print('Input image {}x{}'.format(dim_v, dim_h))
        dim_v = dim_v if self.keep_v else dim_v // 2
        dim_h = dim_h // 2
        if print_network:
            print(' Conv2d > {}x{} {}ch'.format(dim_v,dim_h,channels))
        self.input = rm.Conv2d(
            channels, filter=first_filter, padding=2, 
            stride=(1, 2) if self.keep_v else 2)
        if self.dropout:
            if print_network:
                print(' Dropout')
        for _ in range(blocks):
            t_params, channels = self.denseblock(
                    dim_v=dim_v,
                    dim_h=dim_h,
                    input_channels=channels,
            )
            parameters += t_params
            dim_v = dim_v if self.keep_v else dim_v // 2
            dim_h = (dim_h+1) // 2
        self.hidden = rm.Sequential(parameters)
        nb_parameters = dim_v * dim_h * channels
        self.cnn_channels = channels
        self.nb_parameters = nb_parameters
        self.output_shape = batch_size, channels, dim_v, dim_h
        if print_network:
            print(' Flatten {} params'.format(nb_parameters))
        if check_network:
            self.forward(
                np.zeros((
                    batch_size, 
                    input_shape[0], 
                    input_shape[1],
                    input_shape[2])),
                print_parameter = True
            )

    def denseblock(self,
        dim_v = 8,
        dim_h = 8,
        input_channels=10,
        dropout=False,
        out_ch=0.5,
        ):
        parameters = []
        c = input_channels
        print('-> {}'.format(c))
        for _ in range(self.depth):
            c += self.growth_rate
            print('Batch Normalize')
            parameters.append(rm.BatchNormalize())
            print(' Conv2d > {}x{} {}ch'.format(
                dim_v, dim_h, self.growth_rate 
            ))
            parameters.append(rm.Conv2d(
                self.growth_rate, filter=3, padding=(1,1)
            ))
            if self.dropout:
                print(' Dropout')
        c = int(c*out_ch) if isinstance(out_ch, float) \
            else out_ch 
        print('*Conv2d > {}x{} {}ch'.format(
            dim_v, dim_h, c 
        ))
        parameters.append(rm.Conv2d(
            c, filter=1
        ))
        if self.dropout:
            print(' Dropout')
        print(' Average Pooling')
        print('<- {}'.format(c))
        return parameters, c

    def forward(self, x, print_parameter=False):
        hidden = self.input(x)
        if print_parameter:
            print('{}'.format('-'*20)) 
            print('check network')
            print(x.shape)
            print('{}'.format('-'*20)) 
        if self.dropout:
            hidden = rm.dropout(hidden)
        hidden = rm.max_pool2d(hidden, stride=1, padding=1)
        if print_parameter:
            print(hidden.shape)
            print('{}'.format('-'*20)) 
        layers = self.hidden._layers
        blocks = self.blocks if isinstance(self.blocks, int) else len(self.blocks)
        for i in range(blocks):
            offset = i*(self.depth*2+1)
            for j in range(self.depth):
                sub = rm.leaky_relu(layers[offset+2*j](hidden))
                if print_parameter:
                    print('{}.{} b {}'.format(i,j,sub.shape))
                sub = layers[offset+2*j+1](sub)
                if print_parameter:
                    print('{}.{} + {}'.format(i,j,sub.shape))
                if self.dropout:
                    sub = rm.dropout(sub)
                hidden = rm.concat(hidden, sub)
                if print_parameter:
                    print('{}.{} = {}'.format(i,j,hidden.shape))
            offset = (i+1)*(self.depth*2+1)-1
            hidden = layers[offset](hidden)
            if print_parameter:
                print('{}.{} * {}'.format(i,j,hidden.shape))
            if self.dropout:
                if print_parameter:
                    print('dropout')
                hidden = rm.dropout(hidden)
            hidden = rm.average_pool2d(hidden, padding=1,
                stride=(1, 2) if self.keep_v else 2)
            if print_parameter:
                print('{}.{} @ {}'.format(i,j,hidden.shape))
                print('{}'.format('-'*20)) 
        x = rm.flatten(hidden)
        if print_parameter:
            print('  >>>  {} prameters'.format(x.shape))
        return x

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