"""
@file layers.py

Miscellaneous helper Torch layers
"""
import torch
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, stochastic=True):
        """
        Gaussian sample layer consisting of 2 simple linear layers.
        Can choose whether to fix the variance or let it be learned (training instability has been shown when learning).

        :param in_dim: input dimension (often a flattened latent embedding from a CNN)
        :param out_dim: output dimension
        :param fix_variance: whether to set the log-variance as a constant 0.1
        """
        super(Gaussian, self).__init__()
        self.stochastic = stochastic

        self.mu = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, out_dim)
        )

        self.var = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, out_dim)
        )

        nn.init.zeros_(self.var[-1].weight)

    def reparameterization(self, mu, logvar):
        """ Reparameterization trick to get a sample from the output distribution """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + (noise * std)
        return z

    def forward(self, x):
        """
        Forward function of the Gaussian layer. Handles getting the distributional parameters and sampling a vector
        :param x: input vector [BatchSize, InputDim]
        """
        mu = self.mu(x)

        if self.stochastic is True:
            logvar = self.var(x)

            # Reparameterize and sample
            z = self.reparameterization(mu, logvar)
        else:
            logvar = None
            z = mu

        return mu, logvar, z


class Flatten(nn.Module):
    def forward(self, input):
        """
        Handles flattening a Tensor within a nn.Sequential Block

        :param input: Torch object to flatten
        """
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        """
        Handles unflattening a vector into a 4D vector in a nn.Sequential Block

        :param w: width of the unflattened image vector
        """
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)
