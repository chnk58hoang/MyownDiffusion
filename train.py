from modules.diffusion import GaussianDiffusion
from modules.unet import UNet_conditional
from torch.utils.data import DataLoader
from dataset.dataset import CustomCifar10
from engine.engine import train,Trainer
from utils import plot_images
from torch import nn
import torch
import argparse
import logging



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_dir",type=str)
    parser.add_argument("--save_path",type=str)
    parser.add_argument("--lr", type=float,default=1e-2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_scheduler", type=str, default="linear")
    args = parser.parse_args()
    

    #define the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define the dataset cifar10
    train_dataset = CustomCifar10(data_dir=args.train_dir)

    # define the dataloader

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)

    # define the unet model
    unet_model = UNet_conditional(num_classes=10)

    # define the diffusion
    diffusion = GaussianDiffusion(image_size=args.image_size,num_steps=args.num_steps)

    # define the loss function'
    loss_fn = nn.MSELoss()

    # define the optimizer and the lr scheduler
    optimizer = torch.optim.AdamW(params=unet_model.parameters(),lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=2)

    # define the trainer
    trainer = Trainer(lr_scheduler=lr_scheduler,save_path=args.save_path)


    for epoch in range(args.epoch):
        logging.INFO(f'Epoch: {epoch}/{args.epoch}')
        train_loss = train(train_dataloader,diffusion,unet_model,loss_fn,optimizer,device)
        trainer(train_loss,unet_model,epoch,optimizer)


        if epoch % 10 == 0:
            logging.INFO(f'Start sampling new images')
            labels = torch.arange(10).to(device)
            new_imgs = diffusion.reverse_sampling(unet_model,10,labels)
            plot_images(new_imgs)

        if trainer.stop:
            break




    

