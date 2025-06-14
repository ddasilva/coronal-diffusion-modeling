import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CoronalFieldDatasetHDF
from models import DiffusionModel
from tqdm import tqdm


# Hyperparameters
input_dim = 8372
hidden_dim = 8372
output_dim = 8372
batch_size = 16
epochs = 5
learning_rate = 0.0001
num_workers = 0
out_path_template = "diffusion_model_%d.pth"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/attention")


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

    for batch_idx, (batch, radio_flux) in progress_bar:
        # Calculate noise level for the current epoch
        inputs = batch.to(device)  # Move inputs to GPU if available
        noise_level = torch.rand(inputs.shape[0], 1, device=device)

        radio_flux = radio_flux.to(device)  # Move radio flux to GPU if available

        mean = inputs.mean(dim=0)
        std = inputs.std(dim=0)
        std = torch.maximum(torch.ones_like(std), std)  # Ensure std is not zero
        inputs = (inputs - mean) / std  # Normalize inputs

        optimizer.zero_grad()
        outputs = model(inputs, noise_level, radio_flux)
        loss = criterion(outputs, inputs)  # Train to reconstruct original inputs
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        #scheduler.step()
        epoch_loss += loss.item()

        # Log batch loss to TensorBoard
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
        for batch_idx, (test_batch, radio_flux) in enumerate(test_dataloader):
            test_inputs = test_batch.to(device)
            noise_level = torch.rand(test_inputs.shape[0], 1, device=device)

            radio_flux = radio_flux.to(device)
            mean = test_inputs.mean(dim=0)
            std = test_inputs.std(dim=0)
            std = torch.maximum(torch.ones_like(std), std)  # Ensure std is not zero
            test_inputs = (test_inputs - mean) / std  # Normalize inputs

            test_outputs = model(
                test_inputs, noise_level=noise_level, radio_flux=radio_flux
            )  # Use current noise level during testing
            test_loss += criterion(test_outputs, test_inputs).item()

    test_loss /= len(test_dataloader)
    writer.add_scalar("Validation / Epoch Loss", test_loss, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}")

    # Save the model
    out_path = out_path_template % epoch + 1
    torch.save(model.state_dict(), out_path)
    print('Model checkpoint:', out_path)

# Close the TensorBoard writer
writer.close()
