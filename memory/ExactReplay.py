import torch
import numpy as np

from memory._Memory import Memory


class ExactReplay(Memory):
    def __init__(self, args):
        super().__init__(args)

        # Dictionary of previous tasks and exemplars
        self.tasks = dict()

    def task_update(self):
        pass

    def batch_update(self, images, labels, task_counter=None):
        pass

    def epoch_update(self, _, task_id, model=None):
        # Get full dataset for this task and build psuedo-labels
        images = model.trainer.train_dataloader.dataset.datasets.images.to(f"cuda:{self.args.devices[0]}")
        labels = torch.full([images.shape[0], 1], fill_value=task_id, device=f"cuda:{self.args.devices[0]}")

        # Assign to the dictionary
        self.tasks[task_id] = {
            'images': images,
            'labels': labels
        }

    def get_batch(self):
        # Get proportion of tasks by batch size
        task_size = (self.args.batch_size // 2) // (len(list(self.tasks.keys())))

        # Build outputs of each task
        images, domains, labels = [], [], []
        for task_id in self.tasks.keys():
            sample_indices = np.random.choice(range(self.tasks[task_id]['images'].shape[0]), task_size + self.args.domain_size, replace=False)

            images.append(self.tasks[task_id]['images'][sample_indices][self.args.domain_size:])
            labels.append(self.tasks[task_id]['labels'][sample_indices][self.args.domain_size:])
            domains.append(self.tasks[task_id]['images'][sample_indices][:self.args.domain_size].unsqueeze(0).repeat(task_size, 1, 1, 1, 1))

        # Stack together
        images = torch.vstack(images)
        domains = torch.vstack(domains)
        labels = torch.vstack(labels)
        return images, domains, labels
