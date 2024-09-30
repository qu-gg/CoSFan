"""
@file CommonMetaDynamics.py

A common class that each meta latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
Has a testing step for holdout steps that handles all metric calculations and visualizations.
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning

from utils import metrics
from utils.plotting import show_images
from utils.utils import get_memory, CosineAnnealingWarmRestartsWithDecayAndLinearWarmup
from models.CommonComponents import LocalNormLatentStateEncoder, LocalNormEmissionDecoder, LatentStateEncoder, EmissionDecoder


class LatentMetaDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """ Generic training and testing boilerplate for the dynamics models """
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # Encoder + Decoder
        if self.args.task_setting == "stationary":
            self.encoder = LatentStateEncoder(args)
            self.decoder = EmissionDecoder(args)
        else:
            self.encoder = LocalNormLatentStateEncoder(args)
            self.decoder = LocalNormEmissionDecoder(args)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

        # Update modulator
        self.update_modulator = 1

        # Memory-based component
        self.memory = get_memory(args.memory_name)(args) if args.memory_name != "naive" else None

        # General trackers
        self.n_updates = 0
        self.task_steps = 0
        self.task_counter = -1
        self.task_boundary = False
        self.old_task = False

        # Accumulation of outputs over the logging interval
        self.outputs = list()

    def forward(self, x, D, labels, generation_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    def configure_stationary_optimizer(self):
        """ Configure the stationary setting optimizer, which is AdamW with CosineAnnealing """
        # Define optimizer
        optim = torch.optim.AdamW(list(self.parameters()), lr=self.args.learning_rate)

        # Define step optimizer
        optim_scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optim,
            T_0=5000, 
            T_mult=1,
            eta_min=self.args.learning_rate * 1e-2,
            warmup_steps=200,
            decay=0.9
        )
        
        scheduler = {
            'scheduler': optim_scheduler,
            'interval': 'step'
        }
        return [optim], [scheduler]

    def configure_continual_optimizer(self):
        """ Configure the continual setting optimizer, which has an update and adapt period """
        # Maintain optimizer and LR scheduler over .fit() calls
        if self.task_counter > -1 and self.task_boundary is False and self.args.known_boundary is False:
            optim = self.optimizers().optimizer
            
            if self.args.task_setting == "continual" and self.args.memory_name != "active":
                scheduler = {
                    'scheduler': self.lr_schedulers(),
                    'interval': 'step'
                }
                return [optim], [scheduler]
            else:
                return [optim]

        # Simple catch for first time setting up the optimizer
        if self.task_counter == -1:
            self.task_counter += 1

        # Define optimizer
        optim = torch.optim.AdamW(list(self.parameters()), lr=self.args.learning_rate)

        # Define step optimizer
        if self.args.task_setting == "continual" and self.args.memory_name != "active":
            optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[self.args.num_updates_steps], gamma=0.0001)
            scheduler = {
                'scheduler': optim_scheduler,
                'interval': 'step'
            }
            return [optim], [scheduler]
        else:
            return [optim]

    def configure_optimizers(self):
        """ Optimizer and LR scheduler based on the task setting - stationary or continual"""
        if self.args.task_setting == "continual":
            return self.configure_continual_optimizer()
        else:
            return self.configure_stationary_optimizer()

    def reset_state(self, images=None):
        """ Handles resetting the optimizer/LR scheduler when a new task is detected by the model """
        # Assign a new optimizer and scheduler
        self.trainer.strategy.setup_optimizers(self.trainer)

        # Perform the memory update and update the memory's logger
        if self.memory is not None:
            self.memory.task_update()
            self.memory.update_logger(self.logger)

        # Reset the task boundary flag
        self.task_steps = 0
        self.task_counter += 1
        self.task_boundary = False

    def on_train_start(self):
        """ Boilerplate experiment logging setup pre-training """
        # Get total number of parameters for the model and save
        self.log("total_num_parameters", float(sum(p.numel() for p in self.parameters() if p.requires_grad)), prog_bar=False)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.logger.log_dir}/images/"):
            os.mkdir(f"{self.logger.log_dir}/images/")

    def get_step_outputs(self, batch, generation_len, train=True):
        """ Handles processing a batch and getting model predictions """
        # Get batch
        images, domains, states, domain_state, labels = batch
        images = images[:, :generation_len]
        domains = domains[:, :, :generation_len]

        # Assign the previous images as domains if Train
        if self.args.task_setting == "continual" and train is True and self.n_updates > 0:
            domains = self.previous_domains

        # Set current batch to previous
        if train is True and self.n_updates >= 0:
            previous_indices = np.random.choice(range(images.shape[0]), self.args.domain_size, replace=False)
            self.previous_labels = labels[previous_indices]
            self.previous_domains = images[previous_indices].unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)

        # Get memory batch, added only for training
        if self.memory is not None and train is True and (self.task_counter > 0 or (self.args.memory_name == "active" and self.n_updates >= self.args.num_task_steps)):
            memory_images, memory_domains, memory_labels = self.memory.get_batch()
            images = torch.vstack((images[:self.args.batch_size // 2], memory_images))
            labels = torch.vstack((labels[:self.args.batch_size // 2], memory_labels))
            
            if self.args.memory_name != "active":
                domains = torch.vstack((domains[:self.args.batch_size // 2], memory_domains))

        # Get predictions
        preds, zt = self(images, domains, labels, generation_len)
        return images, domains, states, labels, preds, zt

    def get_step_losses(self, images, domains, preds, labels, train=True):
        """ For a given batch, compute all potential loss terms from components """
        # Reconstruction loss for the sequence and z0
        likelihoods = self.reconstruction_loss(preds, images)
        likelihood = likelihoods.reshape([likelihoods.shape[0] * likelihoods.shape[1], -1]).sum([-1]).mean()

        # Get the loss modulation on continual tasks
        if self.args.task_setting == "continual":
            self.update_modulator = 1 - torch.exp(-self.args.betas.um * likelihood.detach())
        else:
            self.update_modulator = 1

        # Get the likelihood on new task samples only
        new_likelihood = likelihoods[:self.args.batch_size // 2]
        new_likelihood = new_likelihood.reshape([new_likelihood.shape[0] * new_likelihood.shape[1], -1]).sum([-1]).mean()

        # Initial encoder loss, KL[q(z_K|x_0:K) || p(z_K)]
        klz = self.args.betas.z0 * self.encoder.kl_z_term()

        # Get the loss terms from the specific latent dynamics loss
        model_specific_loss = self.model_specific_loss(images, domains, preds)

        # KMEANs loss
        cluster_loss = 0.0
        if self.task_counter > 0 and self.args.memory_name == "cluster":
            dists = torch.sum(torch.square(self.dynamics_func.embeddings[self.args.batch_size // 2:].unsqueeze(1) - self.memory.cluster_means), dim=2)
            assignments = torch.argmin(dists, dim=1).reshape([-1, 1])
            cluster_loss = self.args.betas.cluster * dists[:, assignments].mean()
            self.log("cluster_loss", cluster_loss, prog_bar=True)

        # Get the thresholding metric on the current new data for task-switch checking
        if self.args.known_boundary is False and self.args.boundary_detection is True and self.task_steps > (self.args.num_updates_steps + 1):
            if torch.abs(new_likelihood.detach() - self.previous_likelihood) > self.args.task_threshold:
                with open(f"{self.logger.log_dir}/log.txt", 'a') as f:
                    f.write(f"=> Out-of-distribution Likelihood found for batch: {torch.abs(new_likelihood.detach() - self.previous_likelihood):0.4f} > {self.args.task_threshold:0.4f}\n")
                    f.write(f"=> New task flagged, updating to new task ID {self.task_counter + 1}...\n")
                self.task_boundary = True
                self.reset_state(images[:self.args.batch_size // 2])

        # While still in the locally iid update phase, just assume all samples are the same and update the buffer
        elif self.memory is not None and train is True and self.old_task is False:
            self.memory.batch_update(images[:self.args.batch_size // 2], labels[:self.args.batch_size // 2], self.task_counter + 1)

        # Set previous loss to compare in next batch
        if self.task_steps >= self.args.num_updates_steps:
            self.previous_likelihood = new_likelihood

        # Return all loss terms
        return likelihood, klz, model_specific_loss + cluster_loss

    def get_metrics(self, outputs, setting):
        """ Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard """
        # Convert outputs to Tensors and then Numpy arrays
        images = torch.vstack([out["images"] for out in outputs]).cpu().numpy()
        preds = torch.vstack([out["preds"] for out in outputs]).cpu().numpy()

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(images, preds, args=self.args, setting=setting)[1]
        return out_metrics

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Reshuffle context/query sets
        self.trainer.train_dataloader.dataset.datasets.split()

        # Get outputs and calculate losses
        images, domains, _, labels, preds, _ = self.get_step_outputs(batch, self.args.gen_len)
        likelihood, klz, model_specific_loss = self.get_step_losses(images, domains, preds, labels)

        # Modulate total loss
        loss = self.update_modulator * (likelihood + klz + model_specific_loss)

        # Build the full loss
        self.log_dict({"likelihood": likelihood, "klz_loss": klz, "um": self.update_modulator}, prog_bar=True)

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        # self.outputs.append({"labels": labels.detach().cpu(), "preds": preds.detach().cpu(), "images": images.detach().cpu()})
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """ Every N  Steps, perform the SOM optimization """
        # Check if we're done updating the model in this task.
        if self.memory is not None and self.task_steps == self.args.num_updates_steps and self.old_task is False:
            print("\n=> Doing memory update and switching to fast-evaluation...")
            self.memory.epoch_update(self.logger, self.task_counter, self)

        # Log metrics over saved batches on the specified interval
        if batch_idx % self.args.log_interval == 0 and batch_idx != 0:
            # Model-specific plots
            self.model_specific_plotting(self.logger.log_dir, self.outputs)

            # Wipe the saved batch outputs
            self.outputs = list()

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        self.trainer.test_dataloaders[0].dataset.split()

        # Get model outputs from batch
        images, _, _, labels, preds, _ = self.get_step_outputs(batch, self.args.gen_len, train=False)

        # Return output dictionary
        out = dict()
        for key, item in zip(["labels", "preds", "images"], [labels, preds, images]):
            out[key] = item.detach().cpu().numpy()

        # Add meta-embeddings if it is a meta-model
        if self.args.meta is True:
            out["embeddings"] = self.dynamics_func.embeddings.detach().cpu().numpy()

            # Add meta-embedding distributional parameters if it is a stochastic model
            if self.args.stochastic is True:
                out["embeddings_mu"] = self.dynamics_func.embeddings_mu.detach().cpu().numpy()
                out["embeddings_var"] = self.dynamics_func.embeddings_var.detach().cpu().numpy()
        return out

    def test_epoch_end(self, batch_outputs):
        """ For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder """
        # Set up output path and create dir
        output_path = f"{self.logger.log_dir}/test_{self.args.split}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Stack all output types and convert to numpy
        outputs = dict()
        for key in batch_outputs[0].keys():
            stack_method = np.concatenate if key == "som_assignments" else np.vstack
            outputs[key] = stack_method([output[key] for output in batch_outputs])

        # Save to files
        if self.args.save_files is True:
            for key in outputs.keys():
                np.save(f"{output_path}/test_{self.args.split}_{key}.npy", outputs[key])

        # Iterate through each metric function and add to a dictionary
        print("\n=> getting metrics...")
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_results, metric_mean, metric_std = metric_function(outputs["images"], outputs["preds"], args=self.args, setting='test')
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.args.split}_metrics.json", 'w') as f:
            json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.args.dataset}_excel.txt", 'w') as f:
            for metric in self.args.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")

        # Show side-by-side reconstructions
        show_images(outputs["images"][:10], outputs["preds"][:10], f"{output_path}/test_{self.args.split}_examples.png", num_out=5)
