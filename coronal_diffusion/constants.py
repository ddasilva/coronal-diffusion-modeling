import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timesteps = 1000
beta1 = 1e-4
beta2 = 0.02

# construct noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1
