import numpy as np
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, lr_scheduler, patience=5, save_path='checkpoints/best_model.pth', best_loss=float('inf')):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = best_loss
        self.counter = 0
        self.min_delta = 1e-3
        self.stop = False

    def __call__(self, current_valid_loss, model, epoch, optimizer):
        if self.best_val_loss - current_valid_loss > self.min_delta:
            print(f'Loss improved from {self.best_val_loss} to {current_valid_loss}!')
            self.best_val_loss = current_valid_loss
            self.counter = 0

            print('Saving best model ...')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.save_path)

        else:
            print(f'Loss did not improve from {self.best_val_loss}! Counter {self.counter} of {self.patience}.')
            self.lr_scheduler.step(current_valid_loss)

def train(dataset,dataloader,diffusion,unet_model,loss_fn,optimizer,device):
    train_loss = 0.0
    unet_model = unet_model.to(device)
    unet_model.train()
    for batch_idx,data in tqdm(enumerate(dataloader),total=int(len(dataset)/dataloader.batch_size)):
        images = data[0].to(device)
        labels = data[1].to(device)

        if np.random.random() < 0.1:
            labels = None

        t = diffusion.sample_timestep(images.size(0)).to(device)
        x_t, noise = diffusion.forward_diffusion(images, t)
        predicted_noise = unet_model(x_t, t, labels)
        loss = loss_fn(predicted_noise, noise)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


    





