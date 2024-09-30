import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Buffer of previous samples
        self.buffer = dict()

        # Buffer to hold any new samples that come in over a batch
        self.current_task_indices = None
        self.current_task_data = {
            "images": None,
            "labels": None
        }

        # Number of examples seen
        self.age = 0

    def update_logger(self, logger):
        self.logger = logger

    def save_reservoir(self):
        pass

    def task_update(self):
        # Wipe the accumulated last-task
        self.current_task_indices = None
        self.current_task_data["images"] = None
        self.current_task_data["labels"] = None

    def epoch_update(self, logger, task_id, dynamics_func=None):
        pass

    def batch_update(self, images, labels, task_counter=None):
        pass

    def get_batch(self):
        pass
