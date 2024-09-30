import torch
import functorch
import numpy as np
import torch.nn as nn

from torch.optim import Optimizer
from utils.layers import Gaussian
from models.CommonComponents import LocalDomainEncoder
from models.CommonTraining import LatentMetaDynamicsModel
from torch.distributions import Normal, kl_divergence as kl
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


class BGD(Optimizer):
    """Implements BGD.
    A simple usage of BGD would be:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.randomize_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggregate_grads()
        optimizer.step()
    """

    def __init__(self, params, std_init=0.02, mean_eta=1, mc_iters=5):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        :param std_init: Initialization value for STD parameter
        :param mean_eta: Eta value
        :param mc_iters: Number of Monte Carlo iteration. Used for correctness check.
                         Use None to disable the check.
        """
        super(BGD, self).__init__(params, defaults={})
        assert mc_iters is None or (type(mc_iters) == int and mc_iters > 0), "mc_iters should be positive int or None."
        self.std_init = std_init
        self.mean_eta = mean_eta
        self.mc_iters = mc_iters

        # Initialize mu (mean_param) and sigma (std_param)
        for group in self.param_groups:
            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"

            # group['params'][0] is the weights
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"

            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)

            # Dummy LR for PytorchLightning tracking
            group["lr"] = 1e-3
        self._init_accumulators()

    def get_mc_iters(self):
        return self.mc_iters

    def _init_accumulators(self):
        self.mc_iters_taken = 0
        for group in self.param_groups:
            group["eps"] = None
            group["grad_mul_eps_sum"] = torch.zeros_like(group["params"][0].data).cuda()
            group["grad_sum"] = torch.zeros_like(group["params"][0].data).cuda()

    def randomize_weights(self, force_std=-1):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), 1).cuda()

            # Reparameterization trick (here we set the weights to their randomized value):
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        self.mc_iters_taken += 1
        groups_cnt = 0
        for group in self.param_groups:
            if group["params"][0].grad is None:
                continue
            assert group["eps"] is not None, "Must randomize weights before using aggregate_grads"
            groups_cnt += 1
            grad = torch.clamp(group["params"][0].grad.data, min=-5, max=5).mul(batch_size)
            group["grad_sum"].add_(grad)
            group["grad_mul_eps_sum"].add_(grad.mul(group["eps"]))
            group["eps"] = None
        assert groups_cnt > 0, "Called aggregate_grads, but all gradients were None. Make sure you called .backward()"

    def step(self, closure=None, print_std=False):
        """
        Updates the learned mean and STD.
        :return:
        """
        self.mc_iters_taken = self.mc_iters
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]

            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)

            # Update mean and STD params
            mean.add_(-std.pow(2).mul(e_grad).mul(self.mean_eta))
            sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))

        self.randomize_weights(force_std=0)
        self._init_accumulators()


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
        self.domain_encoder = LocalDomainEncoder(args, args.gen_len)
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


class FeedForwardBGD(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

        # We do manual optimization for BGD since we need aggregation on MC
        self.automatic_optimization = False

    def configure_optimizers(self):
        """ BGD optimizer over parameters """
        # Build the BGD optimizer over all parameters
        params = [{'params': params} for l, (name, params) in enumerate(self.named_parameters())]
        optim = BGD(params, mc_iters=1)

        # Define step optimizer
        dynamics_lr = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[self.args.num_updates_steps], gamma=1e-8)

        # Explicit dictionary to state how often to ping the scheduler
        scheduler = {
            'scheduler': dynamics_lr,
            'interval': 'step'
        }

        return [optim], [scheduler]

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

    def training_step(self, batch, batch_idx):
        """
        A simple usage of BGD would be:
        for samples, labels in batches:
            for mc_iter in range(mc_iters):
                optimizer.randomize_weights()
                output = model.forward(samples)
                loss = cirterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.aggregate_grads()
            optimizer.step()
        """

        """ Training step, getting loss and returning it to optimizer """
        # Reshuffle context/query sets
        self.trainer.train_dataloader.dataset.datasets.split()

        # Get batch
        images, domains, states, domain_state, labels = batch
        images = images[:, :self.args.gen_len]
        domains = domains[:, :, :self.args.gen_len]

        # Get pseudo-labels
        pseudo_labels = torch.full_like(labels, fill_value=self.task_counter)

        # Assign the previous images as domains if Train
        if self.n_updates > 0:
            domains = self.previous_domains

        # Set current batch to previous
        if self.n_updates >= 0:
            previous_indices = np.random.choice(range(images.shape[0]), self.args.domain_size, replace=False)
            self.previous_labels = pseudo_labels[previous_indices]
            self.previous_domains = images[previous_indices].unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)

        for mc_iter in range(5):
            # Randomize weights for Bayesian stuff
            self.optimizers().randomize_weights()

            # Get predictions for this instantiation
            preds, zt = self(images, domains, pseudo_labels, self.args.gen_len)

            # Get model loss terms for the step
            likelihood, klz, model_specific_loss = self.get_step_losses(images, domains, preds, pseudo_labels, train=True)

            # Modulate total loss
            loss = likelihood + klz + model_specific_loss

            # Get gradients and aggregate on step
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().aggregate_grads(preds.shape[0])

        self.optimizers().step()

        # Log the last likelihood
        self.log_dict({"likelihood": likelihood}, prog_bar=True)

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        self.trainer.test_dataloaders[0].dataset.split()

        # Get batch
        images, domains, states, domain_state, true_labels = batch
        images = images[:, :self.args.gen_len]
        domains = domains[:, :, :self.args.gen_len]

        # Assign task psuedo label based on the current task counter
        pseudo_labels = torch.full_like(true_labels, fill_value=self.task_counter)

        # Get predictions over N MC steps
        self.optimizers().randomize_weights()

        # Get predictions for this instantiation
        preds, zt = self(images, domains, pseudo_labels, self.args.gen_len)

        # Return output dictionary
        out = dict()
        for key, item in zip(["labels", "preds", "images"], [true_labels, preds, images]):
            out[key] = item.detach().cpu().numpy()
        return out

    def model_specific_loss(self, x, domain, train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
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
