"""
@file main_continual.py

Main entrypoint for training the CL methods on a random sequence of tasks.
"""
import os
import json
import torch
import hydra
import random
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader import SSMDataModule
from utils.utils import get_model, flatten_cfg
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def plot_continual_metrics(cfg, logger):
    """ Handles plotting a continual performance plot of each unique dynamic over the task numbers for every metric """
    # Get the performance metrics across tasks
    task_performances = dict()
    for task_id in np.unique(cfg.task_ids):
        task_performances[f"task_{task_id}"] = dict()
        for metric in cfg['metrics']:
            task_performances[f"task_{task_id}"][f'{metric}_mean'] = [np.nan for _ in range(len(cfg['task_ids']))]
            task_performances[f"task_{task_id}"][f'{metric}_std'] = [np.nan for _ in range(len(cfg['task_ids']))]

    for idx in range(len(cfg.task_ids)):
        for task_idx in range(len(cfg.task_ids[:idx + 1])):
            true_task_id = cfg.task_ids[task_idx]

            try:
                task_metrics = json.load(
                    open(f"{logger.log_dir}/task_{idx}/test_{task_idx}/test_{task_idx}_metrics.json")
                )

                for metric in cfg['metrics']:
                    task_performances[f"task_{true_task_id}"][f'{metric}_mean'][idx] = task_metrics[f'{metric}_mean']
                    task_performances[f"task_{true_task_id}"][f'{metric}_std'][idx] = task_metrics[f'{metric}_std']

            except Exception as e:
                continue

    def plot_metric(metric_name):
        """ Handles plotting single metric plot """
        plt.rcParams['figure.figsize'] = (10, 5)
        fig, ax = plt.subplots()

        # Plot the performances over tasks over time
        markers = ['o', 'v', '^', '<', '>', 's', '8', 'p', 'o', 'v', '^', '<', '>', 's', '8', 'p']
        dynamics_labels = ['Gravity N', 'Gravity SW', 'Gravity SE',
                           'Pendulum 2G', 'Pendulum 3G', 'Pendulum 4G',
                           'Mass Spring 1K', 'Mass Spring 2K', 'Mass Spring 3K',
                           'Two Body 1G', 'Two Body 2G', 'Two Body 3G']
        handles = []
        for task_id in np.unique(cfg.task_ids):
            task_id = int(task_id)
            plt.plot(range(len(cfg.task_ids)), task_performances[f"task_{task_id}"][f'{metric_name}_mean'], label=f"task_{task_id}", color=cfg.colors[task_id])
            plt.scatter(range(len(cfg.task_ids)), task_performances[f"task_{task_id}"][f'{metric_name}_mean'], marker=markers[task_id], c=cfg.colors[task_id])
            handles.append(mlines.Line2D([], [], marker=markers[task_id], linestyle='None', markersize=10, color=cfg.colors[task_id], label=dynamics_labels[task_id]))

        plt.legend(
            handles=handles,
            loc="lower center",
            ncol=1,
            bbox_to_anchor=(1.13, 0.125),
            fontsize=11
        )

        # Set horizontal gridlines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

        # Set labels
        plt.xticks(ticks=range(len(cfg['task_ids'])), labels=range(len(cfg['task_ids'])), weight='bold')
        ax.set_ylabel(f"{metric_name.upper()}", labelpad=10, weight='bold', fontsize=12)
        ax.set_xlabel('Task #', labelpad=10, weight='bold', fontsize=12)
        ax.set_title(f"Model {metric_name.upper()} Performance Over Tasks", weight='bold', fontsize=15)

        plt.tight_layout()
        plt.savefig(f"{logger.log_dir}/temporal_{metric_name}_performance.png")
        plt.close()

    # Plot each metric in the config
    for metric in cfg.metrics:
        try:
            plot_metric(metric)
        except Exception as e:
            print(f"=> Exception {e} for {metric}, skipping...")

    # Save task performances to a text file
    json.dump(task_performances, fp=open(f"{logger.log_dir}/temporal_metrics.json", 'w'), indent=4)


@hydra.main(version_base="1.3", config_path="configs", config_name="continual")
def main(cfg: DictConfig):
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(cfg.seed, workers=True)
    random.seed(123123)

    # Disable logging for true runs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)

    # Enable fp16 training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # Limit number of CPU workers
    torch.set_num_threads(8)

    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)

    # Shuffle task ids
    dynamic_tasks = cfg.task_ids
    np.random.shuffle(dynamic_tasks)
    dynamic_tasks = dynamic_tasks[:6]
    cfg.task_ids = cfg.task_ids + dynamic_tasks
    np.random.shuffle(cfg.task_ids)

    # Make a consistent color theme across dynamic groups
    cfg.colors = []
    current_color = None
    for task_id in range(12):
        if task_id % 3 == 0:
            current_color = next(plt.gca()._get_lines.prop_cycler)['color']

        cfg.colors.append(current_color)

    print(f"=> Sequence of Tasks: {cfg.task_ids}")
    print(f"=> Sequence of Colors: {cfg.colors}")

    # Build datasets based on tasks
    datamodules = dict()
    for task_id in cfg.task_ids:
        if task_id not in datamodules.keys():
            datamodules[task_id] = SSMDataModule(cfg, task_ids=[task_id])
            print(f"=> Task {task_id}")
            print(f"=> Dataset 'train' shape: {datamodules[task_id].train_dataloader().dataset.images.shape}")
            print(f"=> Dataset 'val' shape: {datamodules[task_id].val_dataloader().dataset.images.shape}")

    # Initialize model
    model = get_model(cfg.model)(cfg)

    # Set up the logger if its train
    logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}")

    # Defining the Trainer
    trainer = pytorch_lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=0,
        max_steps=0,
        gradient_clip_val=cfg.gradient_clip,
        val_check_interval=None,
        num_sanity_val_steps=0,
        inference_mode=cfg.inference_mode
    )
    trainer.callbacks.append(None)

    try:
        # Iterate over tasks, defining the new Task Trainer and evaluating after training
        for idx, task_id in enumerate(cfg.task_ids):
            # Callbacks for logging and tensorboard
            task_logger = pl_loggers.TensorBoardLogger(save_dir=logger.log_dir, name=f"task_{idx}", version='')
            lr_monitor = LearningRateMonitor(logging_interval='step')

            # Checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=f'{task_logger.log_dir}/checkpoints/',
                monitor='val_likelihood',
                filename='step{step:02d}-val_likelihood{val_likelihood:.2f}',
                auto_insert_metric_name=False,
                save_last=True
            )

            # Extend training by another iteration
            trainer.callbacks[-2] = checkpoint_callback
            trainer.callbacks[-1] = lr_monitor
            trainer.logger = task_logger
            trainer.fit_loop.max_epochs += 1
            trainer.fit_loop.max_steps += cfg.num_task_steps * cfg.batch_size

            # Training the model
            trainer.fit(model, datamodules[task_id])
            if model.memory is not None:
                model.memory.update_logger(task_logger)
                model.memory.save_reservoir()

            # Save a checkpoint of the model if automatic optimization is off in submodels
            if model.automatic_optimization is False:
                ckpt_path = os.path.join(model.trainer.log_dir, 'checkpoints', 'last.ckpt')
                model.trainer.save_checkpoint(ckpt_path, weights_only=True)

            # Test on the training set
            cfg.split = "train"
            cfg.task_id = task_id
            trainer.test(model, datamodules[task_id].evaluate_train_dataloader(), ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")

            # Test on all previous tasks and current task, as well as one task forward
            for prev_task_idx, prev_task_id in zip(range(idx + 2), cfg.task_ids[:idx + 2]):
                cfg.split = f"{prev_task_idx}"
                trainer.test(model, datamodules[prev_task_id], ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")

            # If task boundaries are known, then reset the model's optimization state here
            if cfg.known_boundary is True:
                print("=> Known boundary, resetting optimizer state...")
                model.reset_state()

            # Plot continual metrics at this iteration
            plot_continual_metrics(cfg, logger)

            # Remove preds and image npy files
            os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_images.npy' -delete")
            os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_preds.npy' -delete")
            os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_labels.npy' -delete")
    
    except Exception:
        def full_stack():
            import traceback, sys
            exc = sys.exc_info()[0]
            stack = traceback.extract_stack()[:-1]
            if exc is not None:
                del stack[-1]

            trc = 'Traceback (most recent call last):\n'
            stackstr = trc + ''.join(traceback.format_list(stack))
            if exc is not None:
                stackstr += '  ' + traceback.format_exc().lstrip(trc)
            return stackstr
        
        with open(f"{logger.log_dir}/error_log.txt", 'a+') as fp:
            fp.write(full_stack())


if __name__ == '__main__':
    main()
