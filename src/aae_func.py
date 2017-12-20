import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class AAE(rm.Model):
    """
    mode :
    - simple : simple unsupervised clustering
    - incorp_label : semi-supervised learning supporting discriminator
    - supervised : supervised clustering
    - clustering : semi-supervised / unsupervised
    - dim_reduction : semi-supervised / unsupervised

    prior : 
    - normal : normal distribution
    - uniform : uniform distribution
    - gaussians : map to 2d space
    - circle : map to 2d space
    - 10d-gaussians : 10-gaussians map to 2d space
    - predist : map to any 2d grey scale image distribution

    """
    def __init__(
            self, 
            enc, dec, dis,
            latent_dim = 2,
            mode = 'simple',
            label_dim = 0,
            prior = 'normal', 
            prior_dist = None,
            hidden = 1000,
            full_rate=0.1, # 全体の形を重視するかラベル毎を重視するか
            fm_rate=1., # full_rateと同じ目的
        ):
        self.latent_dim = latent_dim
        self.mode = mode
        self.label_dim = label_dim
        self.prior = prior
        self.prior_dist = prior_dist
        self.enc = enc
        self.dec = dec
        self.dis = dis
        self.full_rate = full_rate
        self.fm_rate = fm_rate

    def _set_incorpdist(self, x):
        nb = len(x)
        if self.prior == 'swissroll':
            ""
        else: # gaussians
            div = 10
            smplz = np.zeros((nb, self.latent_dim))
            self.types = random.randint(0, div, size=nb)
            for i in range(div):
                idx = np.where(self.types==i)[0]
                if len(idx) == 0:
                    continue
                rad = i * np.pi / (div/2)
                mean = [np.cos(rad), np.sin(rad)]
                start = [mean[0], mean[1]]
                end = [mean[1]*0.02,mean[0]*0.02]
                mean = [mean[0]*4, mean[1]*4]
                smplz[idx,0], smplz[idx,1] = random.multivariate_normal(
                    mean, [start,end], len(idx)).T
            self.pz = smplz
            if 0: # for debbuging 
                plt.clf()
                plt.scatter(self.pz[:,0], self.pz[:,1], c=self.types)
                plt.show()

    def _set_distribution(self, x):
        nb = len(x)
        if self.prior == 'normal':
            self.pz = random.randn(nb, self.latent_dim)

        elif self.prior == 'uniform':
            size = nb, self.latent_dim
            self.pz = random.uniform(low=-1, high=1, size=size)

        elif self.prior == 'gaussians':
            smplz = random.randn(nb, self.latent_dim)*0.1
            for i in range(self.latent_dim):
                smplz[:,i] += random.randint(1, 3, nb) - 2
            self.pz = smplz

        elif self.prior == 'circle':
            smplz = np.zeros((nb, self.latent_dim))
            for i in range(self.latent_dim):
                temp = random.uniform(low=0., high=np.pi*2, size=nb)
                scale = 4 + np.abs(random.randn(nb))
                if i % 2 == 0:
                    smplz[:,i] = np.cos(temp)*scale
                else:
                    smplz[:,i] = np.sin(temp)*scale
            self.pz = smplz

        elif self.prior == '10d-gaussians':
            ""
        elif self.prior == 'predist':
            idx = random.permutation(len(self.prior_dist))
            self.pz = self.prior_dist[idx[:nb]]

    def _incorp_label(self, x, y, eps=1e-3):
        nb = len(x)

        def comp_fm(_dir, dp, idx_f, idx_m):
            def func(_dir, x):
                return -rm.sum(rm.log(
                        x+eps if _dir=='pos' else 1-x+eps))
            _f = func(_dir, dp[idx_f])/len(idx_f) if len(idx_f) else 0
            _m = func(_dir, dp[idx_m])/len(idx_m) if len(idx_m) else 0
            return self.fm_rate*_f + _m

        # generating pz & check positive judge
        full = int(nb*self.full_rate)
        perm = random.permutation(nb)
        pz_label = np.zeros((nb, self.label_dim+1))
        pz_f_idx, pz_m_idx = perm[full:], perm[:full]
        self.types[pz_f_idx] = self.label_dim
        pz_label[np.arange(nb), self.types] = 1
        self.pz = rm.concat(self.pz, pz_label)
        self.Dpz = self.dis(self.pz)
        self.real = comp_fm(
            'pos', self.Dpz, pz_f_idx, pz_m_idx
        )

        # generating pzx & check negative judge
        if y is None:
            y = np.zeros((nb, self.label_dim+1))
            y[:,-1] = 1
        pzx_f_idx = np.where(y[:,-1]==1)[0]
        pzx_m_idx = np.where(y[:,-1]==0)[0]
        _pzx = rm.concat(self.pzx, y)
        self.Dpzx = self.dis(_pzx)
        self.fake = comp_fm(
            'neg', self.Dpzx, pzx_f_idx, pzx_m_idx
        )
        self.fake2pos = comp_fm(
            'pos', self.Dpzx, pzx_f_idx, pzx_m_idx
        )

    def forward(self, x, y=None, eps=1e-3):
        # x : input data
        # y : one-hot label data for categorical dist. or supporting dis.
        #     empty is not assignment
        # self.qzx : style z
        # self.rep : input data for decoding
        nb = len(x)

        # --- encoding phase --- 
        if 0:
            noise = random.randn(x.size).reshape(nb, x.shape[1])*0.03
            self.pzx = self.enc(x+noise)
        else:
            self.pzx = self.enc(x)

        # --- decoding/reconstruction phase ---
        self.recon = self.dec(self.pzx)

        # --- reguralization phase --- 
        if self.mode == 'incorp_label':
            self._set_incorpdist(x)
        else:
            self._set_distribution(x)
            if self.mode == 'clustering':
                "categorical dist"
            elif self.mode == 'supervised':
                ""
            elif self.mode == 'dim_reduction':
                "" 

        if self.mode == 'incorp_label':
            self._incorp_label(x, y, eps=eps)
        else:
            self.Dpz = self.dis(self.pz)
            self.Dpzx = self.dis(self.pzx)
            self.real = -rm.sum(rm.log(
                self.Dpz + eps
            ))/nb
            self.fake = -rm.sum(rm.log(
                1 - self.Dpzx + eps
            ))/nb
            self.fake2pos = -rm.sum(rm.log(
                self.Dpzx + eps
            ))/nb 

        # --- sumalizing loss ---
        self.gan_loss = self.real + self.fake
        self.reconE = rm.mean_squared_error(self.recon, x)
        self.real_count = (self.Dpz >= 0.5).sum()/nb
        self.fake_count = (self.Dpzx < 0.5).sum()/nb
        self.enc_loss = self.fake2pos

        return self.recon