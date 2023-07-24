from modules.diffusion import GaussianDiffusion
from modules.unet import UNet_conditional
from utils import plot_images
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--weight_path',type=str)
args = parser.parse_args()




n = 10
device = "cuda"
model = UNet_conditional(num_classes=10,device=device).to(device)
ckpt = torch.load()['model_state_dict']
model.load_state_dict(ckpt)
diffusion =GaussianDiffusion(image_size=64, device=device,num_steps=1000)
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.reverse_sampling(model, n, y, cfg_scale=3)
plot_images(x)