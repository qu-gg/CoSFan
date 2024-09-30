import torch
import functorch
import torch.nn as nn

from utils.layers import Gaussian
from models.CommonComponents import LocalDomainEncoder, LatentDomainEncoder
from models.CommonTraining import LatentMetaDynamicsModel
from torch.distributions import Normal, kl_divergence as kl
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


class ODE(nn.Module):
    def __init__(self, args):
        super(ODE, self).__init__()
        self.args = args

        """ Main Network """
        dynamics_network = []
        dynamics_network.extend([
            nn.Linear(args.latent_dim, args.num_hidden),
            nn.SiLU()
        ])

        for _ in range(args.num_layers - 1):
            dynamics_network.extend([
                nn.Linear(args.num_hidden, args.num_hidden),
                nn.SiLU()
            ])

        dynamics_network.extend([nn.Linear(args.num_hidden, args.latent_dim), nn.Tanh()])
        dynamics_network = nn.Sequential(*dynamics_network)
        self.dynamics_network = FunctionalParamVectorWrapper(dynamics_network)

        """ Hyper Network """
        # Domain encoder for z_c
        if args.task_setting == "stationary":
            self.domain_encoder = LatentDomainEncoder(args, args.gen_len)
        elif args.dim == 32:
            self.domain_encoder = LocalDomainEncoder(args, args.gen_len)
        elif args.dim == 64:
            self.domain_encoder = LocalDomainEncoderPDE(args, args.gen_len)
        self.gaussian = Gaussian(args.code_dim, args.code_dim, self.args.stochastic)

        # Hypernetwork going from the embeddings to the full main-network weights
        self.hypernet = nn.Linear(args.code_dim, count_params(dynamics_network))
        nn.init.normal_(self.hypernet.weight, 0, 0.01)
        nn.init.zeros_(self.hypernet.bias)

    def sample_embeddings(self, D):
        """ Given a batch of data points, embed them into their C representations """
        # Reshape to batch get the domain encodings
        domain_size = D.shape[1]
        D = D.reshape([D.shape[0] * domain_size, -1, self.args.dim, self.args.dim])

        # Get domain encoder outputs
        embeddings = self.domain_encoder(D)

        # Reshape to batch and take the average C over each sample
        embeddings = embeddings.view([D.shape[0], domain_size, self.args.code_dim])
        embeddings = embeddings.mean(dim=[1])
        _, _, embeddings = self.gaussian(embeddings)
        return embeddings

    def sample_weights(self, x, D, labels):
        """ Given a batch of data points, embed them into their C representations """
        D = torch.concat((x.unsqueeze(1), D), dim=1)
        domain_size = D.shape[1]

        # Reshape to batch get the domain encodings
        D = D.reshape([D.shape[0] * domain_size, -1, self.args.dim, self.args.dim])

        # Get domain encoder outputs
        self.embeddings = self.domain_encoder(D)

        # Reshape to batch and take the average C over each sample
        self.embeddings = self.embeddings.view([x.shape[0], domain_size, self.args.code_dim])

        # Separate into batch usage and kl usage
        self.embeddings, embeddings_kl = self.embeddings[:, 1:], self.embeddings
        self.embeddings = self.embeddings.mean(dim=[1])
        embeddings_kl = embeddings_kl.mean(dim=[1])

        # From this context set mean, get the distributional parameters
        self.embeddings_mu, self.embeddings_var, self.embeddings = self.gaussian(self.embeddings)
        self.embeddings_kl_mu, self.embeddings_kl_var, _ = self.gaussian(embeddings_kl)

        # Get weight outputs from hypernetwork
        self.params = self.hypernet(self.embeddings)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return z + functorch.vmap(self.dynamics_network)(self.params, z)


class FeedForward(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, D, labels, generation_len):
        # Sample z_init
        z_init = self.encoder(x)

        # Draw weights
        self.dynamics_func.sample_weights(x, D[:, :, :generation_len], labels)

        # Evaluate forward over timestep
        z_cur = z_init
        zts = [z_init]
        for _ in range(generation_len - 1):
            z_cur = self.dynamics_func(None, z_cur)
            zts.append(z_cur)
        
        zt = torch.stack(zts, dim=1)

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_loss(self, x, domain, train=True):
        # Ignore loss if it is a deterministic model
        if self.args.stochastic is False:
            return 0.0

        # Get flattened mus and vars
        embed_mus, embed_vars = self.dynamics_func.embeddings_mu.view([-1]), self.dynamics_func.embeddings_var.view([-1])

        # KL on C with a prior of Normal
        q = Normal(embed_mus, torch.exp(0.5 * embed_vars))
        N = Normal(torch.zeros(len(embed_mus), device=embed_mus.device),
                   torch.ones(len(embed_mus), device=embed_mus.device))

        kl_c_normal = self.args.betas.kl * kl(q, N).view([x.shape[0], -1]).sum([1]).mean()
        self.log("kl_c_normal", kl_c_normal, prog_bar=True)

        # KL on C with a prior of the context set with itself in it
        context_mus, context_vars = self.dynamics_func.embeddings_kl_mu.view([-1]), self.dynamics_func.embeddings_kl_var.view([-1])
        q = Normal(embed_mus, torch.exp(0.5 * embed_vars))
        N = Normal(context_mus, torch.exp(0.5 * context_vars))

        kl_c_context = self.args.betas.kl * kl(q, N).view([x.shape[0], -1]).sum([1]).mean()
        self.log("kl_c_context", kl_c_context, prog_bar=True)

        # Return them as one loss
        return kl_c_normal + kl_c_context
