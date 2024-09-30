"""
@file CommonVAE.py

Holds the encoder/decoder architectures that are shared across the NSSM works
"""
import torch
import torch.nn as nn

from utils.layers import Flatten, Gaussian, UnFlatten
from torch.distributions import Normal, kl_divergence as kl


class LatentStateEncoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        """
        super(LatentStateEncoder, self).__init__()
        self.args = args

        # Encoder, q(z_0 | x_{0:args.z_amort})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(args.z_amort, args.num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(args.num_filters),
            nn.ReLU(),
            nn.Conv2d(args.num_filters, args.num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(args.num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(args.num_filters * 2, args.num_filters * 4, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(args.num_filters * 4),
            nn.ReLU(),
            nn.AvgPool2d(4),
            Flatten()
        )

        self.stochastic_out = Gaussian(args.num_filters * 4, args.latent_dim, self.args.stochastic)
        self.out_act = nn.Tanh()

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        # Return 0 if it is a deterministic model
        if self.args.stochastic is False:
            return 0
        
        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

        q = Normal(mus, torch.exp(0.5 * logvars))
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))

        klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        return klz

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        z0 = self.initial_encoder(x[:, :self.args.z_amort])

        # Apply the Gaussian layer
        self.z_means, self.z_logvs, z0 = self.stochastic_out(z0)
        return self.out_act(z0)


class EmissionDecoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(EmissionDecoder, self).__init__()
        self.args = args

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = args.num_filters * 4 ** 3

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(args.latent_dim, self.conv_dim),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, args.num_filters * 4, kernel_size=4, stride=1, padding=(0, 0)),
            nn.BatchNorm2d(args.num_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters * 4, args.num_filters * 2, kernel_size=5, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(args.num_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters * 2, args.num_filters, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(args.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters, args.num_channels, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        batch_size = zts.shape[0]

        # Flatten to [BS * SeqLen, -1]
        zts = zts.contiguous().view([zts.shape[0] * zts.shape[1], -1])

        # Decode back to image space
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([batch_size, x_rec.shape[0] // batch_size, self.args.dim, self.args.dim])
        return x_rec


class SpatialTemporalBlock(nn.Module):
    def __init__(self, t_in, t_out, n_in, n_out, num_channels, last):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.n_in = n_in
        self.n_out = n_out
        self.num_channels = num_channels
        self.last = last

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=(2, 2))
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.ReLU()
        self.lin_t = nn.Linear(t_in, t_out)

        if last:
            self.act_last = nn.Tanh()

    def forward(self, x):
        B, _, _, H_in, W_in = x.shape
        x = x.contiguous()
        x = x.view(B * self.t_in, self.n_in, H_in, W_in)
        x = self.act(self.bn(self.conv(x)))
        # x = self.act(self.conv(x))
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.view(B, self.t_in, self.n_out, H_out, W_out)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.lin_t(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.last:
            x = self.act_last(x)
            x = x.view(B, -1, H_out, W_out)
        else:
            x = self.act(x)
        return x


class LatentDomainEncoder(nn.Module):
    def __init__(self, args, time_steps):
        """
        Holds the convolutional encoder that takes in a sequence of images and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        """
        super().__init__()
        self.args = args
        self.time_steps = time_steps

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialTemporalBlock(time_steps, time_steps // 2, 1, args.num_filters, args.num_channels, False),
            SpatialTemporalBlock(time_steps // 2, time_steps // 4, args.num_filters, args.num_filters * 2, args.num_channels, False),
            SpatialTemporalBlock(time_steps // 4, 1, args.num_filters * 2, args.num_filters, args.num_channels, True),
            Flatten()
        )

        self.output = nn.Linear(args.num_filters * 4 ** 2, args.code_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.args.num_channels, H, W)
        z = self.encoder(x)
        return self.output(z)


class LocalNormLatentStateEncoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        """
        super(LocalNormLatentStateEncoder, self).__init__()
        self.args = args

        # Encoder, q(z_0 | x_{0:args.z_amort})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(args.z_amort, args.num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.InstanceNorm2d(args.num_filters),
            nn.ReLU(),
            nn.Conv2d(args.num_filters, args.num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.InstanceNorm2d(args.num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(args.num_filters * 2, args.num_filters * 4, kernel_size=5, stride=2, padding=(2, 2)),
            nn.InstanceNorm2d(args.num_filters * 4),
            nn.ReLU(),
            nn.AvgPool2d(4),
            Flatten()
        )

        self.stochastic_out = Gaussian(args.num_filters * 4, args.latent_dim, self.args.stochastic)
        self.out_act = nn.Tanh()

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        # Return 0 if it is a deterministic model
        if self.args.stochastic is False:
            return 0

        # Otherwise get the KL
        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

        q = Normal(mus, torch.exp(0.5 * logvars))
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))

        klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        return klz

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        z0 = self.initial_encoder(x[:, :self.args.z_amort])

        # Apply the Gaussian layer if stochastic version
        self.z_means, self.z_logvs, z0 = self.stochastic_out(z0)
        return self.out_act(z0)


class LocalNormEmissionDecoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(LocalNormEmissionDecoder, self).__init__()
        self.args = args

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = args.num_filters * 4 ** 3

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(args.latent_dim, self.conv_dim),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, args.num_filters * 4, kernel_size=4, stride=1, padding=(0, 0)),
            nn.InstanceNorm2d(args.num_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters * 4, args.num_filters * 2, kernel_size=5, stride=2, padding=(1, 1)),
            nn.InstanceNorm2d(args.num_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters * 2, args.num_filters, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(args.num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(args.num_filters, args.num_channels, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        batch_size = zts.shape[0]

        # Flatten to [BS * SeqLen, -1]
        zts = zts.contiguous().view([zts.shape[0] * zts.shape[1], -1])

        # Decode back to image space
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([batch_size, x_rec.shape[0] // batch_size, self.args.dim, self.args.dim])
        return x_rec


class InstanceSpatialTemporalBlock(nn.Module):
    def __init__(self, t_in, t_out, n_in, n_out, num_channels, last):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.n_in = n_in
        self.n_out = n_out
        self.num_channels = num_channels
        self.last = last

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=(2, 2))
        self.bn = nn.InstanceNorm2d(n_out)
        self.act = nn.ReLU()
        self.lin_t = nn.Linear(t_in, t_out)

        if last:
            self.act_last = nn.Tanh()

    def forward(self, x):
        B, _, _, H_in, W_in = x.shape
        x = x.contiguous()
        x = x.view(B * self.t_in, self.n_in, H_in, W_in)
        x = self.act(self.bn(self.conv(x)))
        # x = self.act(self.conv(x))
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.view(B, self.t_in, self.n_out, H_out, W_out)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.lin_t(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.last:
            x = self.act_last(x)
            x = x.view(B, -1, H_out, W_out)
        else:
            x = self.act(x)
        return x


class LocalDomainEncoder(nn.Module):
    def __init__(self, args, time_steps):
        """
        Holds the convolutional encoder that takes in a sequence of images and
        outputs the domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        """
        super().__init__()
        self.args = args
        self.time_steps = time_steps

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            InstanceSpatialTemporalBlock(time_steps, time_steps // 2, 1, args.num_filters, args.num_channels, False),
            InstanceSpatialTemporalBlock(time_steps // 2, time_steps // 4, args.num_filters, args.num_filters * 2, args.num_channels, False),
            InstanceSpatialTemporalBlock(time_steps // 4, 1, args.num_filters * 2, args.num_filters, args.num_channels, True),
            Flatten()
        )

        self.output = nn.Linear(args.num_filters * 4 ** 2, args.code_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.args.num_channels, H, W)
        z = self.encoder(x)
        return self.output(z)
