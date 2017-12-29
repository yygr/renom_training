import renom as rm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pdb import set_trace
from numpy import random
from time import time
from renom.utility.initializer import Gaussian, Uniform

class Enc(rm.Model):
    def __init__(
        self, pre, latent_shape,
        output_act = None,
        ):
        self.pre = pre
        self.ls = latent_shape
        _net = [rm.Dense(
            np.array(self.ls).prod() if isinstance(self.ls, tuple)
            else self.ls
        )]
        self.output_act = output_act
        self.net = rm.Sequential(_net)
    
    def forward(self, x):
        _nb = len(x)
        _hidden = self.pre(x)
        _hidden = self.net(_hidden)
        if isinstance(self.ls, tuple):
            _out, offset = [], 0
            for j,i in enumerate(self.ls):
                _tmp = _hidden[:,offset:offset+i]
                _a = isinstance(self.output_act, tuple)
                if _a and len(self.output_act)==len(self.ls):
                    if self.output_act[j]:
                        _tmp = self.output_act[j](_tmp)
                _out.append(_tmp)
                offset += i
            _hidden = tuple(_out)
        elif self.output_act:
            _hidden = self.output_act(_hidden)
        return _hidden

class AAE(rm.Model):
    """
    mode :
    - simple : simple unsupervised clustering
    - incorp_label : semi-supervised learning supporting discriminator
    - supervised : supervised clustering
    - clustering : semi-supervised / unsupervised clustering
    - reduction : semi-supervised / unsupervised dimension reduction

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
            enc_base, dec,
            batch_size,
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
        self.batch_size = batch_size
        self.prior = prior
        self.prior_dist = prior_dist
        self.full_rate = full_rate
        self.fm_rate = fm_rate

        if self.mode=='clustering' or self.mode=='reduction':
            self.enc = Enc(enc_base, (latent_dim, label_dim),
            output_act=(None, rm.softmax))
        else:
            self.enc = Enc(enc_base, latent_dim)
        self.dec = dec
        self.dis = rm.Sequential([
            rm.Dense(hidden), rm.LeakyRelu(),
            rm.Dense(hidden), rm.LeakyRelu(),
            rm.Dense(1), rm.Sigmoid()
        ])
        if self.mode=='clustering' or self.mode=='reduction':
            self.cds = rm.Sequential([
                # xxx rm.BatchNormalize(), 
                # Disの最初にBNは配置してはだめ
                rm.Dense(hidden), rm.LeakyRelu(),
                rm.BatchNormalize(),
                rm.Dense(hidden), rm.LeakyRelu(),
                #rm.BatchNormalize(),
                rm.Dense(1), rm.Sigmoid()
            ])

    def _set_incorpdist(self, x):
        nb = len(x)
        div = self.label_dim
        smplz = np.zeros((nb, self.latent_dim))
        self.types = random.randint(0, div, size=nb)
        for i in range(div):
            idx = np.where(self.types==i)[0]
            nb_idx = len(idx)
            if nb_idx == 0:
                continue
            if self.prior == 'swissroll':
                u = ((random.uniform(size=nb_idx)+i)/div)**0.5
                r = u*(np.pi)#**1.1)
                rad = r*(np.pi)#**1.1)
                r += random.randn(nb_idx)*0.01
                smplz[idx,0] = r * np.cos(rad) * 2
                smplz[idx,1] = r * np.sin(rad) * 2
            else: # gaussians
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
        self.pz = self.pz.astype('float32')

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

        # generating qzx & check negative judge
        if y is None:
            y = np.zeros((nb, self.label_dim+1))
            y[:,-1] = 1
        qzx_f_idx = np.where(y[:,-1]==1)[0]
        qzx_m_idx = np.where(y[:,-1]==0)[0]
        _qzx = rm.concat(self.qzx, y)
        self.Dqzx = self.dis(_qzx)
        self.fake = comp_fm(
            'neg', self.Dqzx, qzx_f_idx, qzx_m_idx
        )
        self.fake2pos = comp_fm(
            'pos', self.Dqzx, qzx_f_idx, qzx_m_idx
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
            self._x = x+noise
        else:
            _x = x
        if self.mode=='clustering' or self.mode=='reduction':
            self.qzx, self.qyx = self.enc(_x)
        else:
            self.qzx = self.enc(_x)

        # --- decoding/reconstruction phase ---
        if self.mode=='clustering' or self.mode=='reduction':
            self.recon = self.dec(rm.concat(self.qzx, self.qyx))
        else:
            self.recon = self.dec(self.qzx)

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
            self.Dqzx = self.dis(self.qzx)
            self.real = -rm.sum(rm.log(
                self.Dpz + eps
            ))/nb
            self.fake = -rm.sum(rm.log(
                1 - self.Dqzx + eps
            ))/nb
            self.fake2pos = -rm.sum(rm.log(
                self.Dqzx + eps
            ))/nb 
        if self.mode=='clustering' or self.mode=='reduction':
            _idx = np.where(y.sum(1)==1)[0]
            idx_ = np.where(y.sum(1)==0)[0]
            if len(_idx) > 0:
                self.Cy = self.cds(y)
                self.Cqyx = self.cds(self.qyx)
                self.Creal = -rm.sum(rm.log(
                    self.Cy[_idx] + eps
                ))/len(_idx)
                if 0:
                    self.Cfake = -rm.sum(rm.log(
                        1 - self.Cqyx[_idx] + eps
                    ))/len(_idx)
                else:
                    self.Cfake = -rm.sum(rm.log(
                        1 - self.Cqyx + eps
                    ))/nb
                self.Cfake2 = -rm.sum(rm.log(
                    self.Cqyx[_idx] + eps
                ))/len(_idx)
            else:
                self.Cfake = rm.Variable(0)
                self.Creal = rm.Variable(0)
                self.Cfake2 = rm.Variable(0)

        # --- sumalizing loss ---
        self.gan_loss = self.real + self.fake
        if self.mode=='clustering':
            if len(_idx) > 0:
                self.reconE = rm.mean_squared_error(
                    self.recon[idx_], x[idx_])
            else:
                self.reconE = rm.mean_squared_error(self.recon, x)
        else:
            self.reconE = rm.mean_squared_error(self.recon, x)
        self.real_count = (self.Dpz >= 0.5).sum()/nb
        self.fake_count = (self.Dqzx < 0.5).sum()/nb
        self.enc_loss = self.fake2pos
        if self.mode=='clustering' or self.mode=='reduction':
            if len(_idx) > 0:
                self.Creal_count = (self.Cy[_idx] >= 0.5).sum()/len(_idx)
                self.Cfake_count = (self.Cqyx[_idx] < 0.5).sum()/len(_idx)
            else:
                self.Creal_count = 0
                self.Cfake_count = 0
            self.CganE = self.Creal + self.Cfake
            self.CgenE = self.Cfake2

        return self.recon