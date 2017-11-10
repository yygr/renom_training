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
            #rm.BatchNormalize(),
            rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.Relu(), #rm.Dropout(),
            # xxx rm.BatchNormalize(),
            # Genの最後にBNを配置してはだめ
            rm.Dense(latent_dim, initializer=Uniform())
        ])
        self.dec = rm.Sequential([
            #rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            rm.Dense(28*28), rm.Sigmoid()
        ])
        self.dis = rm.Sequential([
            # xxx rm.BatchNormalize(), 
            # Disの最初にBNは配置してはだめ
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            rm.Dense(hidden), rm.LeakyRelu(),
            #rm.BatchNormalize(),
            rm.Dense(1), rm.Sigmoid()
        ])
    def forward(self, x, y=None, eps=1e-3):
        nb = len(x)
        if 0:
            noise = np.random.randn(x.size).reshape(nb, x.shape[1])*0.03
            self.pzx = self.enc(x+noise)
        else:
            self.pzx = self.enc(x)
        self.recon = self.dec(self.pzx)
        
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
                start = [mean[0], mean[1]]
                end = [mean[1]*0.02,mean[0]*0.02]
                mean = [mean[0]*4, mean[1]*4]
                smplz[idx,0], smplz[idx,1] = np.random.multivariate_normal(
                    mean, [start,end], len(idx)).T
            self.pz = smplz
            if 0: # for debbuging 
                plt.clf()
                plt.scatter(self.pz[:,0], self.pz[:,1], c=types)
                plt.show()

        elif self.prior == 'predist':
            idx = np.random.permutation(len(self.prior_dist))
            self.pz = self.prior_dist[idx[:nb]]

        # label を埋め込んで使う場合
        if self.mode == 'incorp_label':
            full_rate = 0.1 # 全体の形を重視するかラベル毎を重視するか
            learn_rate = 1. # full_rateと同じ目的
            full = int(nb*full_rate)
            perm = np.random.permutation(nb)
            pz_label = np.zeros((nb, self.label_dim+1))
            pz_f_idx = perm[full:]
            pz_m_idx = perm[:full]
            types[pz_f_idx] = self.label_dim
            pz_label[np.arange(nb), types] = 1
            self.pz = rm.concat(self.pz, pz_label)
            if 0: # for debugging 
                plt.clf()
                plt.scatter(self.pz[:,0], self.pz[:,1], 
                    c=np.argmax(self.pz[:,2:], 1))
                plt.show()
        if self.mode == 'incorp_label':
            if y is None:
                y = np.zeros((nb, self.label_dim+1))
                y[:,-1] = 1
            self.pzx = rm.concat(self.pzx, y)
            pzx_f_idx = np.where(y[:,-1]==1)[0]
            pzx_m_idx = np.where(y[:,-1]==0)[0]
            if 0: # for debugging
                plt.clf()
                plt.scatter(self.pz[:,0], self.pz[:,1], 
                    c=np.argmax(self.pz[:,2:], 1))
                plt.scatter(self.pzx[:self.latent_dim,0], 
                    self.z_mu[:self.latent_dim,1],
                    c=np.argmax(y, 1), alpha=0.2, marker='x')
                print('{}'.format(np.unique(np.argmax(y,1))))
                plt.show()
        self.Dispz = self.dis(self.pz)
        self.Dispzx = self.dis(self.pzx)
        if self.mode == 'incorp_label':
            def separate(nn, idx):
                return nn[idx], len(idx)
            Dispz_f, Dispz_f_s = separate(self.Dispz, pz_f_idx)
            Dispz_m, Dispz_m_s = separate(self.Dispz, pz_m_idx)
            Dispzx_f, Dispzx_f_s = separate(self.Dispzx, pzx_f_idx)
            Dispzx_m, Dispzx_m_s = separate(self.Dispzx, pzx_m_idx)
            self.real_f = -rm.sum(rm.log(Dispz_f+eps))/Dispz_f_s
            self.real_m = -rm.sum(rm.log(Dispz_m+eps))/Dispz_m_s
            self.fake_f = -rm.sum(rm.log(1-Dispzx_f+eps))/Dispzx_f_s
            if Dispz_m_s == 0:
                self.fake_m = 0
                self.enc_loss_m = 0
            else:
                self.fake_m = -rm.sum(rm.log(1-Dispzx_m+eps))/Dispzx_m_s
                self.enc_loss_m = -rm.sum(rm.log(Dispzx_m+eps))/Dispzx_m_s
            self.enc_loss_f = -rm.sum(rm.log(Dispzx_f+eps))/Dispzx_f_s
            self.real_count = (self.Dispz >= 0.5).sum()/nb
            self.fake_count = (self.Dispzx < 0.5).sum()/nb
            self.real = learn_rate*self.real_f + self.real_m
            self.fake = learn_rate*self.fake_f + self.fake_m
            self.gan_loss = self.real + self.fake
            self.enc_loss = learn_rate*self.enc_loss_f + self.enc_loss_m
            self.reconE = rm.mean_squared_error(self.recon, x)
        else:
            self.real_count = (self.Dispz >= 0.5).sum()/nb
            self.fake_count = (self.Dispzx < 0.5).sum()/nb
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