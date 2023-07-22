import pytorch_lightning as pl
from utils import plot_images
import torch
import logging

class CustomCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch % 10 == 0:
            logging.info("Sampling new images")
            labels = torch.arange(10).long().to(pl_module.device_)
            sampled_images = pl_module.diffusion.reverse_sampling(pl_module.unet_model, n=len(labels), labels=labels)
            plot_images(sampled_images)


                
