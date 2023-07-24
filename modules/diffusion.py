import torch
import math
from tqdm import tqdm


class GaussianDiffusion():
    def __init__(self,
                 image_size,
                 num_steps,
                 device,
                 beta_start = 1e-4,
                 beta_end = 0.02,
                 beta_scheduler = "linear",
                 ):
        self.num_steps = num_steps
        self.image_size = image_size
        self.beta_start = beta_start
        self.beta_end = beta_end

        if beta_scheduler == 'linear':
            self.betas = self.linear_beta_scheduler()

        elif beta_scheduler == 'cosine':
            self.betas = self.cosine_beta_scheduler()

        assert f"Not implemented for {beta_scheduler}!"

        self.alphas = 1 - self.betas
        self.alphas_tidle = torch.cumprod(self.alphas, dim=-1)
        self.device = device

    def linear_beta_scheduler(self):
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.num_steps)

    def cosine_beta_scheduler(self, s=0.008):
        betas = []
        for t in range(1, self.num_steps):
            f_t = math.cos((t / self.num_steps + s) / (1 + s) * math.pi / 2) ** 2
            f_tsub1 = math.cos(((t - 1) / self.num_steps + s) / (1 + s) * math.pi / 2) ** 2

            beta_t = min(1 - f_t / f_tsub1, 0.999)
            betas.append(beta_t)

        return torch.tensor(betas)

    def sample_timestep(self, n):
        return torch.randint(0, self.num_steps, size=(n,))

    def forward_diffusion(self, x_0, t):
        eps = torch.randn_like(x_0)
        alpha_tidle_t = self.alphas_tidle[t][:, None, None, None]
        x_t = torch.sqrt(alpha_tidle_t) * x_0 + torch.sqrt(1 - alpha_tidle_t) * eps
        return x_t, eps

    def reverse_sampling(self, model, n, labels, cfg_scale=3):
        x_T = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)

        model.eval()
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.num_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                if i > 1:
                    z = torch.randn_like(x_T)
                else:
                    z = torch.zeros_like(x_T)

                un_cond_noise = model(x_T, t)
                cond_noise = model(x_T, t, labels)

                predicted_noise = torch.lerp(un_cond_noise, cond_noise, cfg_scale)

                alpha_t = self.alphas[t][:, None, None, None]
                alpha_tilde_t = self.alphas_tidle[t][:, None, None, None]
                beta_t = self.betas[t][:, None, None, None]

                x = 1 / torch.sqrt(alpha_t) * (
                        x_T - ((1 - alpha_t) / (torch.sqrt(1 - alpha_tilde_t))) * predicted_noise) + torch.sqrt(
                    beta_t) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
