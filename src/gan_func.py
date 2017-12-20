import renom as rm
import numpy as np
from pdb import set_trace

class GAN(rm.Model):
    def __init__(
        self,
        gen,
        dis,
        gan_mode = 'non-saturating',
        latent_dim = None,
    ):
        self.gan_mode = gan_mode
        self.gen = gen
        self.latent_dim = latent_dim if latent_dim else gen.latent_dim
        self.dis = dis
    def forward(self, x, eps=1.e-6, noise=None):
        nb = len(x)
        noise_rate, noise_sigma = 0, 1
        if noise:
            if isinstance(noise,tuple):
                noise_rate, noise_sigma = noise
            else:
                noise_rate = noise
        noise_rate = noise_rate if noise_rate <= 1 else 1
        noise_rate = noise_rate if 0 <= noise_rate else 0
        noise_sigma = noise_sigma if 0 < noise_sigma else np.abs(noise_sigma)

        size = (nb, self.latent_dim)
        if 1:
            z = np.random.randn(
                np.array(size).prod()
                ).reshape(size).astype('float32')
        else:
            z = np.random.uniform(
                -1, 1, np.array(size).prod()
            ).reshape(size).astype('float32')
        self.xz = self.gen(z)


        if noise_rate > 0:
            noised = x.copy()
            _noise_idx = np.random.permutation(nb)
            _noise_idx = _noise_idx[np.random.permutation(int(nb*noise_rate))]
            _noise = np.random.randn(len(_noise_idx))*noise_sigma
            noised[_noise_idx] += _noise.reshape(-1, 1)
            self.Disx = self.dis(noised)
        else:
            self.Disx = self.dis(x)
        self.Disxz = self.dis(self.xz)

        self.real_cost = - rm.sum(
            rm.log(self.Disx + eps))/nb
        self.fake_cost = - rm.sum(
            rm.log(1 - self.Disxz + eps))/nb
        self.GAN_loss = self.real_cost + self.fake_cost

        self.real_count = (self.Disx >= 0.5).sum()/nb
        self.fake_count = (self.Disxz < 0.5).sum()/nb

        if self.gan_mode == 'minmax':
            self.gen_loss = - self.GAN_loss
        elif self.gan_mode == 'non-saturating':
            self.gen_loss = - rm.sum(rm.log(self.Disxz + eps))/nb
        
        return self.xz