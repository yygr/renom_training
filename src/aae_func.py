import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class AAE(rm.Model):
    def __init__(self, latent_dim, prior='normal', prior_dist=None):
        self.latent_dim = latent_dim
        self.prior = prior
        self.prior_dist = prior_dist
        self.enc = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(200), rm.Relu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.Relu(),
            rm.Dense(latent_dim)
        ])
        self.dec = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(10), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(28*28), rm.Sigmoid()
        ])
        self.dis = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(100), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(200), rm.LeakyRelu(),
            rm.Dense(1), rm.Sigmoid()
        ])
    def forward(self, x, eps=1e-3):
        nb = len(x)
        self.z_mu = self.enc(x)
        self.recon = self.dec(self.z_mu)
        self.reconE = rm.mean_squared_error(self.recon, x)

        if self.prior == 'normal':
            self.pz = np.random.randn(nb, self.latent_dim)
        elif self.prior == 'uniform':
            size = nb, self.latent_dim
            self.pz = np.random.uniform(low=-1, high=1, size=size)
        elif self.prior == 'gaussians':
            smplz = np.random.randn(nb, self.latent_dim)*0.1
            for i in range(self.latent_dim):
                smplz[:,i] += np.random.randint(1, 3, nb) - 2
            self.pz = smplz
        elif self.prior == 'circle':
            smplz = np.zeros((nb, self.latent_dim))
            for i in range(self.latent_dim):
                temp = np.random.uniform(low=0., high=np.pi*2, size=nb)
                scale = 4 + np.abs(np.random.randn(nb))
                if i % 2 == 0:
                    smplz[:,i] = np.cos(temp)*scale
                else:
                    smplz[:,i] = np.sin(temp)*scale
            self.pz = smplz
        elif self.prior == '10d-gaussians':
            div = 10
            smplz = np.zeros((nb, self.latent_dim))
            types = np.random.randint(0, div, size=nb)
            for i in range(div):
                idx = np.where(types==i)[0]
                rad = i * np.pi / (div/2)
                mean = [np.cos(rad), np.sin(rad)]
                start = [mean[0]*0.9, mean[1]*0.9+0.03]
                end = [mean[0]*1.1+0.03, mean[1]*1.1]
                smplz[idx,0], smplz[idx,1] = np.random.multivariate_normal(
                    mean, [start,end], len(idx)).T
            self.pz = smplz
        elif self.prior == 'predist':
            idx = np.random.permutation(len(self.prior_dist))
            self.pz = self.prior_dist[idx[:nb]]
            
        self.Dispz = self.dis(self.pz)
        self.Dispzx = self.dis(self.z_mu)
        self.real = -rm.sum(rm.log(
            self.Dispz + eps
        ))/nb
        self.fake = -rm.sum(rm.log(
            1 - self.Dispzx + eps
        ))/nb
        self.enc_loss = -rm.sum(rm.log(
            self.Dispzx + eps
        ))/nb 
        self.gan_loss = self.real + self.fake
        return self.recon