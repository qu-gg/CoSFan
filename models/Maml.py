import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
from models.CommonTraining import LatentMetaDynamicsModel
from models.CommonComponents import LocalNormLatentStateEncoder, LocalNormEmissionDecoder


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


class Maml(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False

        # Locally-normed encoder/decoders
        self.encoder = LocalNormLatentStateEncoder(args)
        self.decoder = LocalNormEmissionDecoder(args)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def sample_weights_testing(self, images, domains, pseudo_labels):
        # Save a copy of the state dict before the updates
        optim = torch.optim.AdamW(list(self.parameters()), lr=self.args.learning_rate)
        weights_before = deepcopy(self.state_dict())

        # Performing the gradient adaptation over each label set
        for label_idx in torch.unique(pseudo_labels):
            indices = torch.where(label_idx == pseudo_labels)[0]
            i, d = images[indices], domains[indices][0]

            """ Inner step """
            for _ in range(self.args.inner_steps):
                pred, _ = self(d, None, None, self.args.gen_len)
                likelihood = self.reconstruction_loss(pred, d).sum([2, 3]).mean([1]).mean([0])

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

    def fast_weights(self, images, domains, pseudo_labels):
        # Save a copy of the state dict before the updates
        optim = self.optimizers()
        weights_before = deepcopy(self.state_dict())

        # Performing the gradient adaptation over each label set
        grads, likelihoods, new_likelihoods, preds, fast_weights = [], [], [], [], []
        for label_idx in torch.unique(pseudo_labels):
            indices = torch.where(label_idx == pseudo_labels)[0]
            i, d = images[indices], domains[indices][0]

            if len(i.shape) < 4:
                i = i.unsqueeze(0)

            """ Inner step """
            for _ in range(self.args.inner_steps):
                pred, _ = self(d, None, None, self.args.gen_len)
                likelihood = self.reconstruction_loss(pred, d).sum([2, 3]).mean([1]).mean([0])

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

            fast_weights.append(torch.concatenate([p.flatten() for p in self.dynamics_func.parameters()]).unsqueeze(0).repeat(i.shape[0], 1))

            """ Outer step """
            optim.zero_grad(set_to_none=True)
            pred, _ = self(i, d, pseudo_labels, self.args.gen_len)
            likelihood_full = self.reconstruction_loss(pred, i).sum([2, 3]).mean([1])
            likelihood = likelihood_full.mean([0])

            # Get the likelihood on new task samples only
            new_likelihood = likelihood_full[:self.args.batch_size // 2].mean([0])

            # Append and get gradients
            likelihoods.append(likelihood)
            new_likelihoods.append(new_likelihood)
            preds.append(pred)
            self.manual_backward(likelihood)

            # Save grads
            sub_grads = []
            for p in self.parameters():
                if p.grad is not None:
                    sub_grads.append(torch.clamp(p.grad.data, min=-5, max=5))
                else:
                    sub_grads.append(None)

            grads.append(sub_grads)

            # Reload base weights
            self.load_state_dict(weights_before)

        return grads, likelihoods, new_likelihoods, torch.vstack(preds), torch.vstack(fast_weights)

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Reshuffle context/query sets
        self.trainer.train_dataloader.dataset.datasets.split()
        
        # Get batch
        images, domains, states, domain_state, labels = batch
        images = images[:, :self.args.gen_len]
        domains = domains[:, :, :self.args.gen_len]

        # Get pseudo-labels
        labels = torch.full_like(labels, fill_value=self.task_counter)

        # Assign the previous images as domains if Train
        if self.args.task_setting == "continual" and self.n_updates > 0:
            domains = self.previous_domains

        # Set current batch to previous
        if self.n_updates >= 0:
            previous_indices = np.random.choice(range(images.shape[0]), self.args.domain_size, replace=False)
            self.previous_labels = labels[previous_indices]
            self.previous_domains = images[previous_indices].unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)

        # Get memory batch, added only for training
        if self.memory is not None and (self.task_counter > 0 or (self.args.memory_name == "active" and self.n_updates >= self.args.num_task_steps)):
            memory_images, memory_domains, memory_labels = self.memory.get_batch()
            images = torch.vstack((images[:self.args.batch_size // 2], memory_images))
            labels = torch.vstack((labels[:self.args.batch_size // 2], memory_labels))
            
            if self.args.memory_name != "active":
                domains = torch.vstack((domains[:self.args.batch_size // 2], memory_domains))

        """ All sample loop """
        grads, likelihoods, new_likelihoods, preds, fast_weights = self.fast_weights(images, domains, labels)

        """ Outer loop """
        optim = self.optimizers()
        optim.zero_grad()

        # Average across the samples
        likelihood = sum(likelihoods) / len(likelihoods)
        new_likelihood = sum(new_likelihoods) / len(new_likelihoods)
        self.log_dict({"likelihood": likelihood}, prog_bar=True)

        # Get the loss modulation
        self.update_modulator = 1 - torch.exp(-self.args.betas.um * likelihood.detach())

        # Aggregate grads
        agg_grads = [[] for _ in range(len(grads[0]))]
        for grad in grads:
            for sub_idx, sub_grad in enumerate(grad):
                agg_grads[sub_idx].append(sub_grad)

        for grad_idx, grad in enumerate(agg_grads):
            if grad[0] is None:
                agg_grads[grad_idx] = None
            else:
                agg_grads[grad_idx] = torch.stack(grad).mean([0])

        # Reapply grads
        for param_old, grad in zip(self.parameters(), agg_grads):
            if grad is not None:
                param_old.data.grad = self.update_modulator * grad
                param_old.grad = self.update_modulator * grad

        # Take the optimizer step and scheduler step
        optim.step()
        if self.args.memory_name != "active":
            self.lr_schedulers().step()

        # Get the thresholding metric on the current new data for task-switch checking
        if self.args.known_boundary is False and self.args.boundary_detection is True and self.task_steps > (self.args.num_updates_steps + 1):
            if torch.abs(new_likelihood.detach() - self.previous_likelihood) > self.args.task_threshold:
                with open(f"{self.logger.log_dir}/log.txt", 'a') as f:
                    f.write(f"=> Out-of-distribution Likelihood found for batch: {torch.abs(new_likelihood.detach() - self.previous_likelihood):0.4f} > {self.args.task_threshold:0.4f}\n")
                    f.write(f"=> New task flagged, updating to new task ID {self.task_counter + 1}...\n")
                self.task_boundary = True
                self.reset_state(images[:self.args.batch_size // 2])

        # While still in the locally iid update phase, just assume all samples are the same and update the buffer
        elif self.memory is not None and self.old_task is False:
            self.memory.batch_update(images[:self.args.batch_size // 2], labels[:self.args.batch_size // 2], self.task_counter + 1)

        # Set previous loss to compare in next batch
        if self.task_steps >= self.args.num_updates_steps:
            self.previous_likelihood = new_likelihood

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        return {"loss": likelihood}

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        self.trainer.test_dataloaders[0].dataset.split()

        # Get model outputs from batch
        with torch.enable_grad():
            # Get batch
            images, domains, states, domain_state, true_labels = batch
            images = images[:, :self.args.gen_len]
            domains = domains[:, :, :self.args.gen_len]

            # Assign task psuedo label based on the current task counter
            pseudo_labels = torch.full_like(true_labels, fill_value=self.task_counter)

            # Get predictions and fast weights
            _, likelihoods, _, preds, fast_weights = self.fast_weights(images, domains, true_labels)

        # Return output dictionary
        out = dict()
        for key, item in zip(["labels", "preds", "images"], [pseudo_labels, preds, images]):
            out[key] = item.detach().cpu().numpy()

        return out
