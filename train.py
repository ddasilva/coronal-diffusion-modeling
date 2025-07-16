from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CoronalFieldDatasetHDF
from models import DiffusionModel, MagneticModel
from tqdm import tqdm
import json
from io import BytesIO
from PIL import Image
import numpy as np
import visualization_tools as vt
import generator as gen
from constants import ab_t, timesteps


# Hyperparameters
input_dim = 8281
hidden_dim = 8281
output_dim = 8281
batch_size = 128
epochs = 100
learning_rate = 0.0001
num_workers = 0
run_name = "experiment7"

harmonics_lambda = 1
#magnetic_lambda = 1e-5
magnetic_lambda = 0

out_path_template = f"checkpoints/{run_name}_%d.pth"
plot_br = True
plot_field_lines = True

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
inputs_mean_square = torch.tensor(scalers["mean_square"]).float().to(device)

# Model, Loss, Optimizer
model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
magnetic_model = MagneticModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)

#harmonic_weights = 1.0 / (inputs_std + 1e-6)
harmonic_weights = 1.0 / torch.clamp(inputs_mean_square, min=100)
magnetic_weights = torch.tensor(magnetic_model.default_r, device=device) ** 2

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{run_name}")


def harmonics_criterian(pred, target, weights):
    #w = weights / weights.sum()
    #return torch.sum((pred - target) ** 2, dim=1).mean()
    return torch.mean((pred - target) ** 2)


def magnetic_criterion(pred, target, weights):
    w = weights / weights.sum()
    w = w.view(1, -1, 1, 1)
    return torch.sum(w * (pred - target) ** 2, dim=1).mean()


def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None] * x + (1 - ab_t[t, None]) * noise


def unperturb_input(noisey_inputs, t, pred_noise):
    ab = ab_t.sqrt()[t, None]
    one_minus_ab = (1 - ab_t[t, None])
    original_inputs = (noisey_inputs - one_minus_ab * pred_noise) / ab
    return original_inputs


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

    # Training: Loop through batches
    # ---------------------------------------------------------------------------
    for batch_idx, (orig_coeffs, radio_flux) in progress_bar:
        # Move inputs to GPU if available
        orig_coeffs = orig_coeffs.to(device)
        radio_flux = radio_flux.to(device)  

        # Normalize inputs
        inputs = (orig_coeffs - inputs_mean) / inputs_std

        # Calculate Noise
        true_noise = torch.randn_like(inputs)
        t = torch.randint(1, timesteps + 1, (inputs.shape[0],)).to(device) 
        noisey_inputs = perturb_input(inputs, t, true_noise)
        noise_level = t / timesteps

        # Call model, loss, and backpropagation
        optimizer.zero_grad()
        pred_noise = model(noisey_inputs, noise_level=noise_level, radio_flux=radio_flux)

        harmonics_loss = harmonics_criterian(pred_noise, true_noise, harmonic_weights) 
        
        if magnetic_lambda == 0:
            magnetic_loss = torch.tensor(0.0, device=device)
        else:
            pred_coeffs = unperturb_input(noisey_inputs, t, pred_noise) * inputs_std + inputs_mean
            pred_magnetic = magnetic_model(pred_coeffs, potential=True)
            target_magnetic = magnetic_model(orig_coeffs, potential=True)
            magnetic_loss = magnetic_criterion(pred_magnetic, target_magnetic, magnetic_weights) 

        loss = harmonics_lambda * harmonics_loss + magnetic_lambda * magnetic_loss

        # Backpropagation
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
            harmonics_lambda * harmonics_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        writer.add_scalar(
            "Training / Magnetogram Loss",
            magnetic_lambda * magnetic_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        # Update progress bar with current loss
        progress_bar.set_postfix(
            loss=loss.item(),
            harmonic_loss=harmonics_lambda*harmonics_loss.item(), 
            mag_loss=magnetic_lambda*magnetic_loss.item()
        )
        
    epoch_loss /= len(train_dataloader)  # Average loss for the epoch

    # Log epoch loss to TensorBoard
    writer.add_scalar("Epoch / Training Loss", epoch_loss, epoch)

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")

    # Plot some sample images
    # ---------------------------------------------------------------------------
    if plot_br:
        print('Plotting magnetogram...')

        for radio_flux, subtitle in zip([0, 1], ["Solar Minimum", "Solar Maximum"]):
            with torch.no_grad():
                G, H = gen.sample(model=model, radio_flux=radio_flux)
            
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
                epoch + 1,
            )

    if plot_field_lines:
        print('Plotting field lines...')

        for radio_flux, subtitle in zip([0, 1], ["Solar Minimum", "Solar Maximum"]):
            with torch.no_grad():
                G, H = gen.sample(model=model, radio_flux=radio_flux)
            
            vis = vt.SHVisualizer(G, H)
            fig = vis.visualize_field_lines(r=1.1, grid_density=40)

            buff = BytesIO()
            fig.write_image(buff, width=800, height=600)   
            buff.seek(0)
            image = np.array(Image.open(buff))

            writer.add_image(
                "Field Lines / (" + subtitle + ")",
                image.transpose(2, 0, 1),
                epoch + 1,
            )
    
    # Evaluate on test set
    # ---------------------------------------------------------------------------
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (orig_coeffs, radio_flux) in enumerate(test_dataloader):
            orig_coeffs = orig_coeffs.to(device)
            radio_flux = radio_flux.to(device)

            inputs = (orig_coeffs - inputs_mean) / inputs_std  # Normalize inputs

            true_noise = torch.randn_like(inputs)
            t = torch.randint(1, timesteps + 1, (inputs.shape[0],)).to(device) 
            noisey_inputs = perturb_input(inputs, t, true_noise)

            pred_noise = model(noisey_inputs, noise_level=t/timesteps, radio_flux=radio_flux) 
            harmonics_loss = harmonics_criterian(pred_noise, true_noise, harmonic_weights).item()

            if magnetic_lambda == 0:
                magnetic_loss = torch.tensor(0.0, device=device)
            else:
                pred_coeffs = unperturb_input(noisey_inputs, t, pred_noise) * inputs_std + inputs_mean
                pred_magnetic = magnetic_model(pred_coeffs, potential=True)
                target_magnetic = magnetic_model(orig_coeffs, potential=True)
                magnetic_loss = magnetic_criterion(pred_magnetic, target_magnetic, magnetic_weights)  # Add br loss
            
            test_loss += harmonics_lambda * harmonics_loss + magnetic_lambda * magnetic_loss

    test_loss /= len(test_dataloader)

    writer.add_scalar("Epoch / Validation Loss", test_loss, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}")

    # Save the model
    out_path = out_path_template % (epoch + 1)
    torch.save(model.state_dict(), out_path)
    print('Model checkpoint:', out_path)

    scheduler.step()  # Step the learning rate scheduler


# Close the TensorBoard writer
writer.close()
