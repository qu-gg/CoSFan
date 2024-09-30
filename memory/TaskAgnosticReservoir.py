import torch
import random
import numpy as np

from memory._Memory import Memory


class TaskAgnosticReservoir(Memory):
    def __init__(self, args):
        super().__init__(args)

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
            for i in range(images.shape[0]):
                self.age += 1
                p = random.randint(0,self.age)  
                if p < self.args.memory_samples:
                    self.buffer["images"][p] = images[i]
                    self.buffer["labels"][p] = labels[i]


    def get_batch(self):
        # Select random indices from the reservoir
        sample_indices = np.random.choice(range(self.buffer["images"].shape[0]), self.args.batch_size // 2, replace=False)

        # Get the corresponding data
        images = self.buffer["images"][sample_indices]
        labels = self.buffer["labels"][sample_indices]
        return images, None, labels

