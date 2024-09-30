import torch
import torch.nn as nn

from copy import deepcopy
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


class MamlAgnostic(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False

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

    def inner_step_batch(self, images):
        """ Handles the inner steps using the full batch of domains given rather than one sample at a time """
        optim = self.optimizers()

        for _ in range(self.args.inner_steps):
            # Inner steps are performed on the given domains
            preds, _ = self(images, None, None, self.args.gen_len)
            likelihood = self.reconstruction_loss(preds, images)

            # Get loss and gradients
            optim.zero_grad(set_to_none=True)
            likelihood = likelihood.reshape([likelihood.shape[0] * likelihood.shape[1], -1]).sum([-1]).mean()
            self.manual_backward(likelihood)

            # Perform SGD over the dynamics function parameters
            for param in self.dynamics_func.dynamics_network.parameters():
                if param.grad is not None:
                    # Note we clip gradients here
                    grad = torch.clamp(param.grad.data, min=-5, max=5)

                    # SGD update
                    param.data -= self.args.inner_learning_rate * grad

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

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Reshuffle context/query sets
        self.trainer.train_dataloader.dataset.datasets.split()

        # Get batch
        images, _, _, _, true_labels = batch
        images = images[:, :self.args.gen_len]

        # Assign task psuedo label based on the current task counter
        pseudo_labels = torch.full_like(true_labels, fill_value=self.task_counter)

        """ Inner loop """
        weights_before = deepcopy(self.state_dict())
        self.inner_step_batch(images[:self.args.batch_size // 2])

        """ Outer loop """
        # Get memory batch, added only for training
        if self.memory is not None and self.n_updates >= self.args.num_task_steps:
            memory_images, _, memory_pseudo_labels = self.memory.get_batch()
            images = torch.vstack((images[:self.args.batch_size // 2], memory_images))
            pseudo_labels = torch.vstack((pseudo_labels[:self.args.batch_size // 2], memory_pseudo_labels))

        # Get meta-batch predictions
        optim = self.optimizers()
        optim.zero_grad(set_to_none=True)
        preds, _ = self(images, None, pseudo_labels, self.args.gen_len)

        # Get model loss terms for the step
        likelihood, klz, _ = self.get_step_losses(images, None, preds, pseudo_labels, train=True)

        # Build the full loss and backprop it
        loss = likelihood + klz
        self.log_dict({"likelihood": likelihood, "klz_loss": klz}, prog_bar=True)
        self.manual_backward(loss)

        # Make a deep copy of these fast-weight gradients
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(torch.clamp(p.grad.data, min=-5, max=5))
            else:
                grads.append(None)

        # Reload the original weights
        optim.zero_grad(set_to_none=True)
        self.load_state_dict(weights_before)

        # Reapply gradients before Adam update
        for param_old, grad in zip(self.parameters(), grads):
            if grad is not None:
                param_old.data.grad = grad
                param_old.grad = grad

        optim.step()

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        self.trainer.test_dataloaders[0].dataset.split()

        with torch.enable_grad():
            # Get batch
            images, domains, _, _, true_labels = batch
            images = images[:, :self.args.gen_len]
            domains = domains[:, :, :self.args.gen_len]

            # Assign task psuedo label based on the current task counter
            pseudo_labels = torch.full_like(true_labels, fill_value=self.task_counter)

            """ Inner loop """
            weights_before = deepcopy(self.state_dict())
            self.inner_step_batch(domains)

        """ Outer loop """
        # Get meta-batch predictions
        optim = self.optimizers()
        optim.zero_grad(set_to_none=True)
        preds, _ = self(images, domains, pseudo_labels, self.args.gen_len)

        # Reload the original weights
        self.load_state_dict(weights_before)

        # Return output dictionary
        out = dict()
        for key, item in zip(["labels", "preds", "images"], [pseudo_labels, preds, images]):
            out[key] = item.detach().cpu().numpy()
        return out
