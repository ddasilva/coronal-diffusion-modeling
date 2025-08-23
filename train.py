import json
from io import BytesIO

from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import numpy as np

from coronal_diffusion.dataset import CoronalFieldDatasetHDF
from coronal_diffusion.models import DiffusionModel, MagneticModel
from coronal_diffusion import visualization_tools as vt
from coronal_diffusion import sampler
from coronal_diffusion import constants

import config


def save_checkpoint(model, config, epoch):
    checkpoint_path_template = f"checkpoints/{config.run_name}_%d.pth"
    out_path = checkpoint_path_template % (epoch + 1)
    torch.save(model.state_dict(), out_path)
    print('Model checkpoint:', out_path)


def get_dataloaders(config):
    train_dataset = CoronalFieldDatasetHDF(config.train_dataset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    test_dataset = CoronalFieldDatasetHDF(config.test_dataset_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    return train_dataloader, test_dataloader
    

def get_scalers(config):
    with open(config.scalers_path) as fh:
        scalers = json.load(fh)

    scalers_mean = torch.tensor(scalers["mean"]).float().to(constants.device)
    scalers_std = torch.tensor(scalers["std"]).float().to(constants.device)

    return scalers_mean, scalers_std
    
        
def harmonics_criterian(pred, target, weights):
    w = weights / weights.sum()
    return torch.sum(w * (pred - target) ** 2, dim=1).mean()


def magnetic_criterion(pred, target, weights):
    w = weights / weights.sum()
    w = w.view(1, -1, 1, 1)
    return torch.sum(w * (pred - target) ** 2, dim=1).mean()


def perturb_input(x, t, noise):
    return constants.ab_t.sqrt()[t, None] * x + (1 - constants.ab_t[t, None]).sqrt() * noise


def unperturb_input(noisey_inputs, t, pred_noise):
    ab = constants.ab_t.sqrt()[t, None]
    one_minus_ab = (1 - constants.ab_t[t, None]).sqrt()
    original_inputs = (noisey_inputs - one_minus_ab * pred_noise) / constants.ab
    return original_inputs


def do_br_plot(config, model, epoch, writer):
    print('Plotting Br (magnetogram)...')

    tasks = [
        (0.0, "Solar Minimum", config.seed_helper_min),
        (1.0, "Solar Maximum", config.seed_helper_max),
    ]
    
    for radio_flux, subtitle, seed_helper in tasks:
        with torch.no_grad():
            G, H = sampler.sample(model=model, radio_flux=radio_flux, seed_helper=seed_helper)
            
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


def do_field_line_plot(config, model, epoch, writer):
    print('Plotting field lines...')

    tasks = [
        (0.0, "Solar Minimum", config.seed_helper_min),
        (1.0, "Solar Maximum", config.seed_helper_max),
    ]
    
    for radio_flux, subtitle, seed_helper in tasks:
        with torch.no_grad():
            G, H = sampler.sample(model=model, radio_flux=radio_flux, seed_helper=seed_helper)
            
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


def do_test_loop(
        model, magnetic_model, test_dataloader, scalers_mean, scalers_std,
        epoch, harmonics_weights, magnetic_weights, config, writer
):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (orig_coeffs, radio_flux) in enumerate(test_dataloader):
            orig_coeffs = orig_coeffs.to(constants.device)
            radio_flux = radio_flux.to(constants.device)

            inputs = (orig_coeffs - scalers_mean) / scalers_std  # Normalize inputs

            true_noise = torch.randn_like(inputs)
            t = torch.randint(1, constants.timesteps + 1, (inputs.shape[0],)).to(constants.device) 
            noisey_inputs = perturb_input(inputs, t, true_noise)

            pred_noise = model(noisey_inputs, noise_level=t/constants.timesteps, radio_flux=radio_flux) 
            harmonics_loss = harmonics_criterian(pred_noise, true_noise, harmonics_weights).item()

            if config.magnetic_lambda == 0:
                magnetic_loss = torch.tensor(0.0, device=constants.device)
            else:
                pred_coeffs = unperturb_input(noisey_inputs, t, pred_noise) * scalers_std + scalers_mean
                pred_magnetic = magnetic_model(pred_coeffs, potential=True)
                target_magnetic = magnetic_model(orig_coeffs, potential=True)
                magnetic_loss = magnetic_criterion(pred_magnetic, target_magnetic, magnetic_weights)  # Add br loss
            
            test_loss += config.harmonics_lambda * harmonics_loss + config.magnetic_lambda * magnetic_loss

    test_loss /= len(test_dataloader)

    writer.add_scalar("Epoch / Validation Loss", test_loss, epoch)
    print(f"Epoch [{epoch+1}/{config.epochs}], Test Loss: {test_loss:.4f}")

        
def do_train_loop(
        model, magnetic_model, train_dataloader, scalers_mean, scalers_std,
        epoch, harmonics_weights, magnetic_weights, config, writer, optimizer
):
    model.train()
        
    print("Starting epoch", epoch + 1)

    # Initialize progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch+1}/{config.epochs}",
    )

    # Training: Loop through batches
    epoch_loss = 0.0
    
    for batch_idx, (orig_coeffs, radio_flux) in progress_bar:
        # Move inputs to GPU 
        orig_coeffs = orig_coeffs.to(constants.device)
        radio_flux = radio_flux.to(constants.device)  

        # Normalize inputs
        inputs = (orig_coeffs - scalers_mean) / scalers_std

        # Calculate Noise
        true_noise = torch.randn_like(inputs)
        t = torch.randint(1, constants.timesteps + 1, (inputs.shape[0],)).to(constants.device) 
        noisey_inputs = perturb_input(inputs, t, true_noise)
        noise_level = t / constants.timesteps

        # Call model, loss, and backpropagation
        optimizer.zero_grad()
        pred_noise = model(noisey_inputs, noise_level=noise_level, radio_flux=radio_flux)

        harmonics_loss = harmonics_criterian(pred_noise, true_noise, harmonics_weights) 
        
        if config.magnetic_lambda == 0:
            magnetic_loss = torch.tensor(0.0, device=constants.device)
        else:
            pred_coeffs = unperturb_input(noisey_inputs, t, pred_noise) * scalers_std + scalers_mean
            pred_magnetic = magnetic_model(pred_coeffs, potential=True)
            target_magnetic = magnetic_model(orig_coeffs, potential=True)
            magnetic_loss = magnetic_criterion(pred_magnetic, target_magnetic, magnetic_weights) 

        loss = (
            config.harmonics_lambda * harmonics_loss
            + config.magnetic_lambda * magnetic_loss
        )

        # Backpropagation
        loss.backward()

        # Clip gradients to prevent exploding gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            config.harmonics_lambda * harmonics_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        writer.add_scalar(
            "Training / Magnetogram Loss",
            config.magnetic_lambda * magnetic_loss.item(),
            epoch * len(train_dataloader) + batch_idx,
        )

        # Update progress bar with current loss
        progress_bar.set_postfix(
            loss=loss.item(),
            harmonic_loss=config.harmonics_lambda*harmonics_loss.item(), 
            mag_loss=config.magnetic_lambda*magnetic_loss.item()
        )
    
    epoch_loss /= len(train_dataloader)  # Average loss for the epoch

    # Log epoch loss to TensorBoard
    writer.add_scalar("Epoch / Training Loss", epoch_loss, epoch)

    print(f"Epoch [{epoch+1}/{config.epochs}], Training Loss: {epoch_loss:.4f}")


def main():
    # Load dataloaders and scalers
    train_dataloader, test_dataloader = get_dataloaders(config)    
    scalers_mean, scalers_std = get_scalers(config)

    # Setup base denoising diffusion model and the secondary magnetic potential
    # expansion model
    model = DiffusionModel().to(constants.device)
    magnetic_model = MagneticModel().to(constants.device)

    # Setup the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma
    )

    # Set the weights used in the harmonics loss function
    harmonics_weights = torch.ones(*scalers_std.shape, device=constants.device)
    magnetic_weights = torch.tensor(magnetic_model.default_r, device=constants.device) ** 2

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{config.run_name}")

    # Epoch loop
    for epoch in range(config.epochs):
        do_train_loop(
            model, magnetic_model, train_dataloader, scalers_mean, scalers_std, 
            epoch, harmonics_weights, magnetic_weights, config, writer, optimizer,
        )

        do_test_loop(
            model, magnetic_model, test_dataloader, scalers_mean, scalers_std, epoch,
            harmonics_weights, magnetic_weights, config, writer
        )

        save_checkpoint(model, config, epoch)

        scheduler.step()
        
        if config.plot_br:
            do_br_plot(config, model, epoch, writer)

        if config.plot_field_lines:
            do_field_line_plot(config, model, epoch, writer)        

        writer.flush()
            
    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
