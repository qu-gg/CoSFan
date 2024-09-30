"""
@file main_stationary.py

Main entrypoint for training the stationary models over a set of tasks.
"""
import torch
import hydra
import random
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader import SSMDataModule
from utils.utils import get_model, flatten_cfg
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(version_base="1.3", config_path="configs", config_name="stationary")
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

    # Build datasets based on tasks
    dataset = SSMDataModule(cfg, task_ids=range(cfg.num_dynamics))
    print(f"=> Dataset 'train' shape: {dataset.train_dataloader().dataset.images.shape}")
    print(f"=> Dataset 'val' shape: {dataset.val_dataloader().dataset.images.shape}")

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
        val_check_interval=cfg.val_log_interval,
        num_sanity_val_steps=0,
        inference_mode=cfg.inference_mode
    )
    trainer.callbacks.append(None)

    # Callbacks for logging and tensorboard
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{logger.log_dir}/checkpoints/',
        monitor='val_likelihood',
        filename='step{step:02d}-val_likelihood{val_likelihood:.2f}',
        auto_insert_metric_name=False,
        save_last=True
    )

    # Extend training by another iteration
    trainer.callbacks[-2] = checkpoint_callback
    trainer.callbacks[-1] = lr_monitor
    trainer.logger = logger
    trainer.fit_loop.max_epochs += 1
    trainer.fit_loop.max_steps += cfg.num_task_steps * cfg.batch_size

    try:
        # Training the model
        trainer.fit(model, dataset.train_dataloader())

        # Test on the training set
        cfg.split = "train"
        cfg.task_id = 0
        trainer.test(model, dataset.evaluate_train_dataloader(), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")
        
        cfg.split = "test"
        cfg.task_id = 0
        trainer.test(model, dataset.test_dataloader(), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")

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
