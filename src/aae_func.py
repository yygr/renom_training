import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class AAE(rm.Model):
    def __init__(
            self, 
            latent_dim, 
            mode = 'normal',
            label_dim = 0,
            prior = 'normal', 
            prior_dist = None,
            hidden = 1000
        ):
        self.latent_dim = latent_dim
        self.mode = mode
        self.label_dim = label_dim
        self.prior = prior
        self.prior_dist = prior_dist
        self.enc = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
            rm.Dense(latent_dim, initializer=Uniform())
        ])
        self.dec = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            rm.Dense(28*28), rm.Sigmoid()
        ])
        self.dis = rm.Sequential([
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            #rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            #rm.Dense(hidden), rm.LeakyRelu(),
            rm.Dense(1), rm.Sigmoid()
        ])
    def forward(self, x, y=None, eps=1e-3):
        nb = len(x)
        self.z_mu = self.enc(x)
        self.recon = self.dec(self.z_mu)

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
                if len(idx) == 0:
                    continue
                rad = i * np.pi / (div/2)
                mean = [np.cos(rad), np.sin(rad)]
                start = [mean[0]*5, mean[1]*5]
                end = [mean[1]*0.1,mean[0]*0.1]
                mean = [mean[0]*5, mean[1]*5]
                smplz[idx,0], smplz[idx,1] = np.random.multivariate_normal(
                    mean, [start,end], len(idx)).T
            self.pz = smplz
            #plt.scatter(self.pz[:,0], self.pz[:,1], c=types.reshape(-1))
            #plt.show()

        elif self.prior == 'predist':
            idx = np.random.permutation(len(self.prior_dist))
            self.pz = self.prior_dist[idx[:nb]]

        # label を埋め込んで使う場合
        if self.mode == 'incorp_label':
            pz_label = np.zeros((nb, self.label_dim+1))
            types = np.random.randint(0, div+1, size=nb)
            pz_label[np.arange(nb), types] = 1
            perm = np.random.permutation(nb)
            pz_full_mixture_idx = perm[:-nb//5]
            pz_mixture_comp_idx = perm[-nb//5:]
            unlabeled = np.zeros((1, self.label_dim+1))
            unlabeled[0,-1] = 1
            pz_label[pz_full_mixture_idx] = unlabeled
            self.pz = np.c_[self.pz, pz_label]
        if not y is None:
            self.pzx = np.c_[self.z_mu, y]
            pzx_full_mixture_idx = np.where(y[-1]==1)[0]
            pzx_mixture_comp_idx = np.where(y[-1]==0)[0]

        self.Dispz = self.dis(self.pz)
        self.Dispzx = self.dis(self.pzx)
        if 0:#self.mode == 'incorp_label':
            self.real_fm = -rm.sum(rm.log(
                self.Dispz[pz_full_mixture_idx] + eps
            ))/len(pz_full_mixture_idx)
            self.real_mc = -rm.sum(rm.log(
                self.Dispz[pz_mixture_comp_idx] + eps
            ))/len(pz_mixture_comp_idx)
            self.fake_fm = -rm.sum(rm.log(
                1 - self.Dispzx[pzx_full_mixture_idx] + eps
            ))/len(pzx_full_mixture_idx)
            self.fake_mc = -rm.sum(rm.log(
                1 - self.Dispzx[pzx_mixture_comp_idx] + eps
            ))/len(pzx_mixture_comp_idx)
            self.enc_loss_fm = -rm.sum(rm.log(
                self.Dispzx[pzx_full_mixture_idx] + eps
            ))/len(pzx_full_mixture_idx)
            self.enc_loss_mc = -rm.sum(rm.log(
                self.Dispzx[pzx_mixture_comp_idx] + eps
            ))/len(pzx_mixture_comp_idx)
            self.gan_loss_fm = self.real_fm + self.fake_fm
            self.gan_loss_mc = self.real_mc + self.fake_mc
            self.reconE_fm = rm.mean_squared_error(
                self.recon[pzx_full_mixture_idx],
                x[pzx_full_mixture_idx]
            )
            self.reconE_mc = rm.mean_squared_error(
                self.recon[pzx_mixture_comp_idx],
                x[pzx_mixture_comp_idx]
            )
            self.real = self.real_fm + self.real_mc
            self.fake = self.fake_fm + self.fake_mc
            self.gan_loss = self.gan_loss_fm + self.gan_loss_mc
            self.enc_loss = self.enc_loss_fm + self.enc_loss_mc
            self.reconE = self.reconE_fm + self.reconE_mc
        else:
            self.reconE = rm.mean_squared_error(self.recon, x)
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