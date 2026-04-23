import torch

N_CONTEXT = 6    # Number of context items
N_REAL = 12      # Number of ADAPT realizations

# CUDA Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diffusion settings and noise schedule parameters. 
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02

b_t = (beta2 - beta1) * torch.linspace(
    0, 1, timesteps + 1, device=device
) + beta1  # small to big
a_t = 1 - b_t  # big to small
ab_t = torch.cumsum(a_t.log(), dim=0).exp()  # big to small
ab_t[0] = 1
