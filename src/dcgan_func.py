import renom as rm
import numpy as np
from pdb import set_trace

class Gen(rm.Model):
    def __init__(
            self,
            latent_dim = 10,
            output_shape = (28, 28), 
            batch_normal = False,
            dropout = False,
            min_channels = 16,
        ):
        self.batch_normal = batch_normal
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.dropout = dropout
        self.min_channels = min_channels
        print('--- Generator Network ---')
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
            if batch_normal:
                parameters.append(rm.BatchNormalize())
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
        length = len(layers) if not self.batch_normal else len(layers)//2
        for i in range(length):
            if self.batch_normal:
                h = layers[2*i](h)
                h = rm.relu(layers[2*i+1](h))
            else:
                h = rm.relu(layers[i](h))
            #print(h.shape)
        h = self.output(h)
        #return rm.sigmoid(h)
        return rm.tanh(h)

class Dis(rm.Model):
    def __init__(self,):
        channel = 8
        intermidiate_dim = 128
        self.cnn1 = rm.Sequential([
            # 28x28 -> 28x28
            rm.Conv2d(channel=channel,filter=3,stride=1,padding=1),
            rm.LeakyRelu(),
            rm.Dropout(),
            # 28x28 -> 14x14
            rm.Conv2d(channel=channel*2,filter=3,stride=2,padding=1),
            rm.LeakyRelu(),
            rm.Dropout(),
            # 14x14 -> 8x8
            rm.Conv2d(channel=channel*4,filter=3,stride=2,padding=2),
            rm.LeakyRelu(),
            rm.Dropout(),
            # 8x8 -> 4x4
            rm.Conv2d(channel=channel*8,filter=3,stride=2,padding=1),
            rm.LeakyRelu(),
            rm.Dropout(),
        ])
        self.cnn2 = rm.Sequential([
            #rm.Dropout(),
            rm.Flatten(),
            #rm.Dense(intermidiate_dim)
        ])
        self.output = rm.Dense(1)
    def forward(self, x):
        self.lth = self.cnn1(x)
        hidden = self.cnn2(self.lth)
        self.raw_output = self.output(hidden)
        return rm.sigmoid(self.raw_output)

class DCGAN(rm.Model):
    def __init__(self, gen, dis):
        self.gen = gen
        self.dis = dis
        self.latent_dim = gen.latent_dim
    def forward(self, x, eps=1e-3):
        nb = len(x)
        size = (nb, self.latent_dim)
        zp = np.random.randn(np.array(size).prod()).reshape(size).astype('float32')
        self.xp = self.gen(zp)
        self.Dis_xp = self.dis(self.xp)
        self.Dis_xp_is = self.dis.raw_output
        self.Dis_x = self.dis(x)

        self.real_cost = - rm.sum(rm.log(self.Dis_x + eps))/nb
        self.fake_cost = - rm.sum(rm.log(1 - self.Dis_xp + eps))/nb
        self.GAN_loss = self.real_cost + self.fake_cost

        gan_mode = 'non-saturating'
        if gan_mode == 'minmax':
            self.gen_loss = - self.GAN_loss
        elif gan_mode == 'non-saturating':
            self.gen_loss = - rm.sum(rm.log(self.Dis_xp + eps))/nb
        elif gan_mode == 'max-likelihood':
            self.gen_loss = - rm.sum(rm.exp(self.Dis_xp_is))/nb

        return self.GAN_loss
