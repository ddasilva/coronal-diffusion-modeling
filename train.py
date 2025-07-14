from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CoronalFieldDatasetHDF
from models import DiffusionModel, BrModel
from tqdm import tqdm
import json
from io import BytesIO
from PIL import Image
import numpy as np
import visualization_tools as vt
import generator as gen


# Hyperparameters
input_dim = 8281
hidden_dim = 8281
output_dim = 8281
batch_size = 128
epochs = 50
learning_rate = 0.0001
num_workers = 0
run_name = "br-experiment"
br_lambda = 1e-6
out_path_template = f"checkpoints/{run_name}_%d.pth"
include_br = True
include_harmonics = False
plot_br = True
plot_br_freq = 100

# Dataset and DataLoader
train_dataset = CoronalFieldDatasetHDF("training_dataset.h5")
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

test_dataset = CoronalFieldDatasetHDF("test_dataset.h5")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

# Get GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load scalers
with open('scalers.json') as fh:
    scalers = json.load(fh)

inputs_mean = torch.tensor(scalers["mean"]).float().to(device)
inputs_std = torch.tensor(scalers["std"]).float().to(device)
inputs_mean_abs = torch.tensor(scalers["mean_abs"]).float().to(device)

# Model, Loss, Optimizer
model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
br_model = BrModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0)

harmonic_weights = 1.0 / (inputs_std + 1e-6)
br_weights = torch.tensor(br_model.default_r, device=device) ** (-3)  # Inverse cube law for Br

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{run_name}")


def harmonics_criterian(pred, target, weights):
    w = weights / weights.sum()
    return torch.sum(w * (pred - target) ** 2, dim=1).mean()


def br_criterion(pred_br, target_br, br_weights):
    w = br_weights / br_weights.sum()
    w = w.view(1, -1, 1, 1)
    return torch.sum(w * (pred_br - target_br) ** 2, dim=1).mean()


# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    print("Starting epoch", epoch + 1)

    # Initialize progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch+1}/{epochs}",
    )

    for batch_idx, (orig_coeffs, radio_flux) in progress_bar:
        # Move inputs to GPU if available
        orig_coeffs = orig_coeffs.to(device)
        radio_flux = radio_flux.to(device)  

        # Normalize inputs
        inputs = (orig_coeffs - inputs_mean) / inputs_std

        # Calculate Noise
        noise_level = torch.rand(inputs.shape[0], device=device) 
        noise_repeated = noise_level.repeat(inputs.shape[1], 1).T
        true_noise = torch.normal(mean=torch.zeros_like(noise_repeated), std=noise_repeated).to(device)        
        noisey_inputs = inputs + true_noise  # Add noise

        # Call model, loss, and backpropagation
        optimizer.zero_grad()
        pred_noise = model(noisey_inputs, noise_level=noise_level, radio_flux=radio_flux)

        if include_harmonics:
            harmonics_loss = harmonics_criterian(pred_noise, true_noise, harmonic_weights)  # Train to reconstruct noise
        else:
            harmonics_loss = torch.tensor(0)
        
        if include_br:
            pred_coeffs = (orig_coeffs - pred_noise) * inputs_std + inputs_mean
            pred_br = br_model(pred_coeffs)
            target_br = br_model(orig_coeffs)
            br_loss = br_lambda * br_criterion(pred_br, target_br, br_weights)  # Add br loss
        else:
            br_loss = torch.tensor(0)

        if plot_br and batch_idx > 0 and batch_idx % plot_br_freq == 0:
            for radio_flux, subtitle in zip([0, 1], ["Solar Minimum", "Solar Maximum"]):
                with torch.no_grad():
                    G, H = gen.sample(model=model, nsteps=25, radio_flux=radio_flux)
                vis = vt.SHVisualizer(G, H)
                vis.plot_magnetogram()

                buff = BytesIO()
                plt.savefig(buff, format='png', dpi=100)
                plt.close()
                
                buff.seek(0)
                image = np.array(Image.open(buff))

                writer.add_image(
                    "Generated Magnetogram / (" + subtitle + ")",
                    image.transpose(2, 0, 1),
                    epoch * len(train_dataloader) + batch_idx,
                )

        loss = harmonics_loss + br_lambda * br_loss
        
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameter
        optimizer.step()

        # Add loss to running loss
        epoch_loss += loss.item()

        writer.add_scalar(
            "Training / Total Loss",
            loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        writer.add_scalar(
            "Training / Harmonics Loss",
            harmonics_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        writer.add_scalar(
            "Training / Magnetogram Loss",
            br_lambda * br_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item(), harmonic_loss=harmonics_loss.item(), magnetogram_loss=br_lambda*br_loss.item())

    epoch_loss /= len(train_dataloader)  # Average loss for the epoch

    # Log epoch loss to TensorBoard
    writer.add_scalar("Epoch / Training Loss", epoch_loss, epoch)

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, radio_flux) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            radio_flux = radio_flux.to(device)

            noise_level = torch.rand(inputs.shape[0], device=device)
            noise_repeated = noise_level.repeat(inputs.shape[1], 1).T
            true_noise = torch.normal(mean=torch.zeros_like(noise_repeated), std=noise_repeated).to(device)

            inputs = (inputs - inputs_mean) / inputs_std  # Normalize inputs
            noise_inputs = inputs + true_noise  # Add noise to inputs

            pred_noise = model(noise_inputs, noise_level=noise_level, radio_flux=radio_flux) 
            test_loss += harmonics_criterian(pred_noise, true_noise, harmonic_weights).item()

    test_loss /= len(test_dataloader)

    writer.add_scalar("Epoch / Validation Loss", test_loss, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}")

    # Save the model
    out_path = out_path_template % (epoch + 1)
    torch.save(model.state_dict(), out_path)
    print('Model checkpoint:', out_path)

# Close the TensorBoard writer
writer.close()
