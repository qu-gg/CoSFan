import torch
import numpy as np

from memory._Memory import Memory


class TaskAwareReservoir(Memory):
    def __init__(self, args):
        super().__init__(args)

    def epoch_update(self, logger, task_counter, model=None):
        # Update the buffer with the accumulated last-task samples
        self.buffer["images"][self.current_task_indices] = self.current_task_data["images"]
        self.buffer["labels"][self.current_task_indices] = torch.full([self.current_task_data["images"].shape[0], 1], fill_value=task_counter, device=self.args.devices[0])

        # Wipe the accumulated last-task
        self.current_task_indices = None
        self.current_task_data["images"] = None
        self.current_task_data["labels"] = None

        # Print out the current distribution to stdout
        print(f"=> Distribution of labels: {np.unique(self.buffer['labels'].detach().cpu().numpy(), return_counts=True)}")

    def batch_update(self, images, labels, task_counter=None):
        # Initialize vectors at first batch
        if self.age == 0:
            self.buffer["images"] = images[0].unsqueeze(0)
            self.buffer["labels"] = labels[0].unsqueeze(0)
            self.age += 1

        # Just add to buffer if buffer is not filled
        elif self.args.memory_samples - self.age > 0:
            stopping_point = self.args.memory_samples - self.age
            self.buffer["images"] = torch.vstack((self.buffer["images"], images[:stopping_point]))
            self.buffer["labels"] = torch.vstack((self.buffer["labels"], labels[:stopping_point]))
            self.age += self.args.batch_size // 2

        # Otherwise take a proportional sample and add if it is under threshold
        else:
            p = torch.randint(0, self.age, [self.args.batch_size // 2])
            indices = torch.where(p < self.args.memory_samples)[0]
            p = p[indices]

            # If current task buffer is not initialized, just assign values
            if self.current_task_indices is None:
                self.current_task_indices = p
                self.current_task_data["images"] = images[indices]

            # Other append *only* unique indices thus far, skip repeats
            else:
                # Find the non-intersecting set between the already set indices and new ones
                non_insect = p[(p[:, None] != self.current_task_indices).all(dim=1)]

                # Then get those indices from the indice set
                indices_to_use = []
                for non_i in non_insect:
                    indices_to_use.append(np.where(p == non_i)[0][0])
                indices_to_use = np.array(indices_to_use)

                # Add to current task buffer
                self.current_task_indices = torch.concatenate((self.current_task_indices, p[indices_to_use]))
                self.current_task_data["images"] = torch.vstack((self.current_task_data["images"], images[indices_to_use]))

            self.age += self.args.batch_size // 2

    def get_batch(self):
        # Select random indices from the reservoir
        sample_indices = np.random.choice(range(self.buffer["images"].shape[0]), self.args.batch_size // 2, replace=False)

        # Get the corresponding data
        images = self.buffer["images"][sample_indices]
        labels = self.buffer["labels"][sample_indices]
        buffer_labels = self.buffer["labels"].detach().cpu().numpy()

        # Get the domains for the image based on the prototype it is assigned to
        domains = []
        for pid in labels.detach().cpu().numpy():
            domain_indices = np.where(buffer_labels == pid[0])[0]
            domains.append(self.buffer["images"][np.random.choice(domain_indices, self.args.domain_size, replace=True)])

        # Stack the domains
        domains = torch.stack(domains)
        return images, domains, labels

