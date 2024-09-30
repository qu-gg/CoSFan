import torch
import torch.nn as nn

from models.CommonTraining import LatentMetaDynamicsModel


class ODE(nn.Module):
    def __init__(self, args):
        super(ODE, self).__init__()
        self.args = args

        # Build the dynamics network
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
        self.dynamics_network = nn.Sequential(*dynamics_network)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return z + self.dynamics_network(z)


class RGNRes(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, D, labels, generation_len):
        # Sample z_init
        z_init = self.encoder(x)

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
