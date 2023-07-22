from typing import Any, Optional
from modules.diffusion import GaussianDiffusion
from modules.unet import UNet_conditional
from dataset.datamodule import DataModule
from callback.callback import CustomCallback
from torchvision.datasets.cifar import CIFAR10
from torch import nn
import torch
import pytorch_lightning as pl
import numpy as np
import argparse

class Module(pl.LightningModule):
    def __init__(self,
                 diffusion,
                 unet_model,
                 lr):
        super().__init__()
        self.diffusion = diffusion
        self.unet_model = unet_model
        self.loss = nn.MSELoss()
        self.device_ = self.diffusion.device
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.unet_model.parameters(), lr=self.lr)
        return {'optimizer': optimizer}

    def training_step(self, batch):
        images = batch[0].to(self.device_)
        labels = batch[1].to(self.device_)

        if np.random.random() < 0.1:
            labels = None

        t = self.diffusion.sample_timestep(images.size(0))
        x_t, noise = self.diffusion.forward_diffusion(images, t)
        predicted_noise = self.unet_model(x_t, t, labels)
        loss = self.loss(predicted_noise, noise)

        return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_dir",type=str)
    parser.add_argument("--lr", type=float,default=1e-2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_scheduler", type=str, default="linear")
    args = parser.parse_args()
    


    # define the dataset cifar10
    train_dataset = CIFAR10(root=args.train_dir,download=True,train=True)

    # define the lightning datamodule
    data_module = DataModule(train_dataset=train_dataset,valid_dataset=None,batch_size=args.batch_size)

    # define the diffusion process
    gausian_diffusion = GaussianDiffusion(image_size=args.image_size,num_steps=args.num_steps,beta_start=args.beta_start,
                                          beta_end=args.beta_end,beta_scheduler=args.beta_scheduler)
    
    # define the unet_model
    unet_model = UNet_conditional()

    #define the lightning module
    module = Module(diffusion=gausian_diffusion,unet_model=unet_model,lr=args.lr)
    #define the callback

    callback = CustomCallback()

    trainer = pl.Trainer(max_epochs=args.epoch,callbacks=[callback,],accelerator='gpu')

    trainer.fit(model=module,datamodule=data_module)


    

