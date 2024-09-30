"""
@file

"""
import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
from torch.optim import Optimizer
from models.CommonTraining import LatentMetaDynamicsModel


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
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters are optimized
        in the metaspace.
        :param args: script arguments to use for initialization
        """
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


class MamlBGD(LatentMetaDynamicsModel):
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
        return optim

    def sample_weights_testing(self, images, domains, pseudo_labels):
        # Save a copy of the state dict before the updates
        optim = torch.optim.AdamW(list(self.parameters()), lr=self.args.learning_rate)
        weights_before = deepcopy(self.state_dict())

        for _ in range(self.args.inner_steps):
            pred, _ = self(domains[0], None, None, self.args.gen_len)
            likelihood = self.reconstruction_loss(pred, domains[0]).sum([2, 3]).mean([1]).mean([0])

            # Get loss and gradients
            optim.zero_grad(set_to_none=True)
            likelihood.backward()

            # Perform SGD over the dynamics function parameters
            for param in self.dynamics_func.dynamics_network.parameters():
                if param.grad is not None:
                    # Note we clip gradients here
                    grad = torch.clamp(param.grad.data, min=-5, max=5)

                    # SGD update
                    param.data -= self.args.inner_learning_rate * grad

        # Reload base weights
        self.load_state_dict(weights_before)

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
            domains = self.previous_domains[0]

        # Set current batch to previous
        if self.n_updates >= 0:
            previous_indices = np.random.choice(range(images.shape[0]), self.args.domain_size, replace=False)
            self.previous_labels = pseudo_labels[previous_indices]
            self.previous_domains = images[previous_indices].unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)

        for mc_iter in range(5):
            # Randomize weights for Bayesian stuff
            self.optimizers().randomize_weights()

            # Save a copy of the state dict before the updates
            optim = self.optimizers()
            weights_before = deepcopy(self.state_dict())

            """ Inner step """
            for _ in range(self.args.inner_steps):
                pred, _ = self(domains[0], None, None, self.args.gen_len)
                likelihood = self.reconstruction_loss(pred, domains[0]).sum([2, 3]).mean([1]).mean([0])

                # Get loss and gradients
                optim.zero_grad(set_to_none=True)
                self.manual_backward(likelihood)

                # Perform SGD over the dynamics function parameters
                for param in self.dynamics_func.dynamics_network.parameters():
                    if param.grad is not None:
                        # Note we clip gradients here
                        grad = torch.clamp(param.grad.data, min=-5, max=5)

                        # SGD update
                        param.data -= self.args.inner_learning_rate * grad

            """ Outer step """
            optim.zero_grad(set_to_none=True)
            preds, _ = self(images, domains, pseudo_labels, self.args.gen_len)
            likelihood_full = self.reconstruction_loss(pred, images).sum([2, 3]).mean([1])
            likelihood = likelihood_full.mean([0])

            # Append and get gradients
            self.manual_backward(likelihood)
            self.optimizers().aggregate_grads(preds.shape[0])

            # Reload base weights
            self.load_state_dict(weights_before)

        self.optimizers().step()

        # Log the last likelihood
        self.log_dict({"likelihood": likelihood}, prog_bar=True)

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        return {"loss": likelihood}

    def validation_step(self, batch, batch_idx):
        """ Validation step over a single batch """
        # Shuffle context and query sets
        self.trainer.val_dataloaders[0].dataset.split()

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

        # Get model loss terms for the step
        likelihood = self.get_step_losses(images, domains, preds, pseudo_labels, train=True)
        self.log("val_likelihood", likelihood, prog_bar=True)

        # Return outputs as dict
        return {"loss": likelihood, "preds": preds.detach(), "images": images.detach()}

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
