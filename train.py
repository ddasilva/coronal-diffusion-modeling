import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CoronalFieldDatasetHDF
from models import DiffusionModel
from tqdm import tqdm
import json

# Hyperparameters
input_dim = 8281
hidden_dim = 8281
output_dim = 8281
batch_size = 128
epochs = 50
learning_rate = 0.0001
num_workers = 0
run_name = "deeper-mlp-two-hidden-layers"
out_path_template = f"checkpoints/{run_name}_%d.pth"

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
inputs_mean_abs[inputs_mean_abs < 1e-3] = 1e-3 # avoid numerical issues

# Model, Loss, Optimizer
model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
#criterion = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{run_name}")


def criterion(pred, target, weights):
    w = weights / weights.sum()
    return torch.sum(w * (pred - target) ** 2, dim=1).mean()


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

    for batch_idx, (inputs, radio_flux) in progress_bar:
        # Move inputs to GPU if available
        inputs = inputs.to(device)
        radio_flux = radio_flux.to(device)  

        # Normalize inputs
        inputs = (inputs - inputs_mean) / inputs_std

        # Calculate Noise
        noise_level = torch.rand(inputs.shape[0], device=device)
        noise_repeated = noise_level.repeat(inputs.shape[1], 1).T
        true_noise = torch.normal(mean=torch.zeros_like(noise_repeated), std=noise_repeated).to(device)        
        noisey_inputs = inputs + true_noise  # Ndd noise

        # Call model, loss, and backpropagation
        optimizer.zero_grad()
        pred_noise = model(noisey_inputs, noise_level=noise_level, radio_flux=radio_flux)
        loss = criterion(pred_noise, true_noise, inputs_mean_abs)  # Train to reconstruct noise
        loss.backward()

        # Clip gradients to prevent exploding gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameter
        optimizer.step()

        # Add loss to running loss
        epoch_loss += loss.item()

        writer.add_scalar(
            "Training / Batch Loss",
            loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss /= len(train_dataloader)  # Average loss for the epoch

    # Log epoch loss to TensorBoard
    writer.add_scalar("Training / Epoch Loss", epoch_loss, epoch)

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

            test_loss += criterion(pred_noise, true_noise, inputs_mean_abs).item()

    test_loss /= len(test_dataloader)

    writer.add_scalar("Validation / Epoch Loss", test_loss, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}")

    # Save the model
    out_path = out_path_template % (epoch + 1)
    torch.save(model.state_dict(), out_path)
    print('Model checkpoint:', out_path)

# Close the TensorBoard writer
writer.close()
