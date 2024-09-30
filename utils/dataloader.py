"""
@file dataloader.py

Holds the Datasets and Dataloaders
"""
import torch
import numpy as np
import pytorch_lightning

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler


class SSMDataset(Dataset):
    """ Basic Dataset object for the SSM """

    def __init__(self, args, images, labels, states, controls):
        self.args = args
        self.images = images
        self.labels = labels
        self.states = states
        self.controls = controls

        # Build the label indices in this dataset
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Do the first split
        self.split()

    def __len__(self):
        return self.images.shape[0] - (len(self.label_idx.keys()) * self.args.domain_size)

    def __getitem__(self, idx):
        # Get the true idx of the sample
        idx = self.qry_idx[idx]

        # Get the dynamic label
        label = self.labels[idx].item()

        # Build out the query sample
        image_qry = self.images[idx, :]
        state_qry = self.states[idx, :]

        # Get the corresponding context set
        image_spt = self.images[self.spt_idx[label], :]
        state_spt = self.states[self.spt_idx[label], :]
        return image_qry, image_spt, state_qry, state_spt, torch.as_tensor([label])

    def split(self):
        """ Handles generating a new context-query split """
        self.spt_idx = {}
        self.qry_idx = []
        for label_id, samples in self.label_idx.items():
            qry_idx, spt_idx = train_test_split(range(len(samples)), test_size=self.args.domain_size)

            self.spt_idx[label_id] = samples[spt_idx]
            self.qry_idx.extend(samples[qry_idx])

        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


class SSMDataModule(pytorch_lightning.LightningDataModule):
    """ Custom DataModule object that handles preprocessing all sets of data for a given run """
    def __init__(self, args, task_ids):
        super(SSMDataModule, self).__init__()
        self.args = args
        self.task_ids = task_ids

    def make_loader(self, mode="train", evaluation=True, shuffle=True):
        # Iterate over task ids and stack datasets
        image_stack, label_stack, state_stack, control_stack = [], [], [], []
        for task_id in self.task_ids:
            npzfile = np.load(f"data/{self.args.dataset}/{self.args.dataset_ver}{task_id}/{mode}.npz")

            # Load in data sources
            images = npzfile['image']
            labels = npzfile['label'].astype(np.int16)
            states = npzfile['state'].astype(np.float32)[:, :, :2]

            # Load control, if it exists, else make a dummy one
            controls = npzfile['control'] if 'control' in npzfile \
                else np.zeros((images.shape[0], images.shape[1], 1), dtype=np.float32)

            # Modify based on dataset percent
            rand_idx = np.random.choice(range(images.shape[0]), size=int(images.shape[0] * self.args.dataset_percent), replace=False)
            images = images[rand_idx]
            labels = labels[rand_idx]
            states = states[rand_idx]
            controls = controls[rand_idx]

            # Convert to Tensors
            images = torch.from_numpy(images).float()
            labels = torch.from_numpy(labels)
            states = torch.from_numpy(states)
            controls = torch.from_numpy(controls)

            # Append
            image_stack.append(images)
            label_stack.append(labels)
            state_stack.append(states)
            control_stack.append(controls)

        # Stack datasets together
        image_stack = torch.vstack(image_stack)
        label_stack = torch.vstack(label_stack)
        state_stack = torch.vstack(state_stack)
        control_stack = torch.vstack(control_stack)

        # Build dataset and corresponding Dataloader
        dataset = SSMDataset(self.args, image_stack, label_stack, state_stack, control_stack)

        # Build dataloader based on whether it is training or evaluation
        if evaluation is False:
            sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=self.args.num_task_steps * self.args.batch_size)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True)
        return dataloader

    def train_dataloader(self):
        """ Getter function that builds and returns the training dataloader """
        return self.make_loader("train", evaluation=False)

    def evaluate_train_dataloader(self):
        return self.make_loader("train", shuffle=False)

    def val_dataloader(self):
        """ Getter function that builds and returns the validation dataloader """
        return self.make_loader("val", shuffle=False)

    def test_dataloader(self):
        """ Getter function that builds and returns the testing dataloader """
        return self.make_loader("test", shuffle=False)
