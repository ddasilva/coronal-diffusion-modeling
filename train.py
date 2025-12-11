import json
from io import BytesIO

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn as nn

from coronal_diffusion.dataset import CoronalFieldDatasetHDF
from coronal_diffusion.models import DiffusionModel
from coronal_diffusion import visualization_tools as vt
from coronal_diffusion import sampler
from coronal_diffusion import constants
from coronal_diffusion.utils import flat_to_GH

import config


def save_checkpoint(model, config, epoch):
    checkpoint_path_template = f"checkpoints/{config.run_name}_%d.pth"
    out_path = checkpoint_path_template % (epoch + 1)
    torch.save(model.state_dict(), out_path)
    print("Model checkpoint:", out_path)


def get_dataloaders(config):
    train_dataset = CoronalFieldDatasetHDF(config.train_dataset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
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

    scalers_dict = {}
    scalers_dict["std"] = np.array(scalers["std"])
    scalers_dict["unscaled_abs"] = np.array(scalers["unscaled_abs"])

    return scalers_dict


def perturb_input(x, t, noise):
    return (
        constants.ab_t.sqrt()[t, None, None, None] * x
        + (1 - constants.ab_t[t, None, None, None]).sqrt() * noise
    )


def unperturb_input(noisey_inputs, t, pred_noise):
    ab = constants.ab_t.sqrt()[t, None]
    one_minus_ab = (1 - constants.ab_t[t, None]).sqrt()
    original_inputs = (noisey_inputs - one_minus_ab * pred_noise) / ab
    return original_inputs


def do_img_plot(config, model, epoch, writer):
    print("Plotting Magnetic Potential at photosphere (spherical harmonic image)...")

    tasks = [
        (
            0.0,
            "Solar Minimum",
        ),
        (
            1.0,
            "Solar Maximum",
        ),
    ]

    for radio_flux, subtitle in tasks:
        with torch.no_grad():
            img, (G, H) = sampler.sample(model=model, radio_flux=radio_flux)

        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)

        for i in range(img.shape[0]):
            axes.flatten()[i].imshow(img[i])

        buff = BytesIO()
        plt.savefig(buff, format="png", dpi=300)
        plt.close()

        buff.seek(0)
        image = np.array(Image.open(buff))

        writer.add_image(
            "Generated Image / (" + subtitle + ")",
            image.transpose(2, 0, 1),
            epoch + 1,
        )


def do_br_plot(config, model, epoch, writer):
    print("Plotting Br (magnetogram)...")

    tasks = [
        (
            0.0,
            "Solar Minimum",
        ),
        (
            1.0,
            "Solar Maximum",
        ),
    ]

    for radio_flux, subtitle in tasks:
        with torch.no_grad():
            img, (G, H) = sampler.sample(model=model, radio_flux=radio_flux)

        if not np.isfinite([G, H]).all():
            continue

        vis = vt.SHVisualizer(G, H)
        vis.plot_magnetogram()

        buff = BytesIO()
        plt.savefig(buff, format="png", dpi=100)
        plt.close()

        buff.seek(0)
        image = np.array(Image.open(buff))

        writer.add_image(
            "Generated Magnetogram / (" + subtitle + ")",
            image.transpose(2, 0, 1),
            epoch + 1,
        )


def do_field_line_plot(config, model, epoch, writer):
    print("Plotting field lines...")

    tasks = [
        (
            0.0,
            "Solar Minimum",
        ),
        (
            1.0,
            "Solar Maximum",
        ),
    ]

    for radio_flux, subtitle in tasks:
        with torch.no_grad():
            img, (G, H) = sampler.sample(
                model=model,
                radio_flux=radio_flux,
            )

        if not np.isfinite([G, H]).all():
            continue

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


# def do_test_loop(
#         model, magnetic_model, test_dataloader, scalers_mean, scalers['std'],
#         epoch, harmonics_weights, magnetic_weights, config, writer
# ):
#     model.eval()
#     test_loss = 0

#     with torch.no_grad():
#         for batch_idx, (orig_coeffs, radio_flux) in enumerate(test_dataloader):
#             # Break if reached max batch
#             if batch_idx == config.max_test_batches:
#                 break

#             # Move inputs to GPU
#             orig_coeffs = orig_coeffs.to(constants.device)
#             radio_flux = radio_flux.to(constants.device)

#             # Normalize inputs
#             inputs = (orig_coeffs - scalers_mean) / scalers['std']

#             true_noise = config.noise_inflation * torch.randn_like(inputs)
#             t = torch.randint(1, constants.timesteps + 1, (inputs.shape[0],)).to(constants.device)
#             noisey_inputs = perturb_input(inputs, t, true_noise)

#             pred_noise = model(noisey_inputs, noise_level=t/constants.timesteps, radio_flux=radio_flux)

#             harmonics_loss = harmonics_criterian(pred_noise, true_noise)

#             if config.magnetic_lambda == 0:
#                 magnetic_loss = torch.tensor(0.0, device=constants.device)
#             else:
#                 pred_coeffs = unperturb_input(noisey_inputs, t, pred_noise) * scalers['std'] + scalers_mean
#                 pred_magnetic = magnetic_model(pred_coeffs, potential=True)
#                 target_magnetic = magnetic_model(orig_coeffs, potential=True)
#                 magnetic_loss = magnetic_criterion(pred_magnetic, target_magnetic, magnetic_weights)  # Add br loss

#             test_loss += config.harmonics_lambda * harmonics_loss + config.magnetic_lambda * magnetic_loss

#     test_loss /= len(test_dataloader)

#     writer.add_scalar("Epoch / Validation Loss", test_loss, epoch)
#     print(f"Epoch [{epoch+1}/{config.epochs}], Test Loss: {test_loss:.4f}")


def do_train_loop(
    model, train_dataloader, scalers_dict, epoch, config, writer, optimizer
):
    model.train()

    print("Starting training of epoch", epoch + 1)

    # Initialize progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader),
        total=min(config.max_train_batches, len(train_dataloader)),
        desc=f"Epoch {epoch+1}/{config.epochs}",
    )

    # Training: Loop through batches
    epoch_loss = 0.0
    radii = np.arange(
        config.min_radius, config.max_radius + config.res_radius, config.res_radius
    )
    num_batches = 0

    for batch_idx, (orig_coeffs, radio_flux) in progress_bar:
        # Break if reached max batch
        if batch_idx == config.max_train_batches:
            break

        # Move inputs to GPU
        orig_coeffs = orig_coeffs.to(constants.device)
        radio_flux = radio_flux.to(constants.device)

        # Normalize inputs
        img_true = get_potential_images(model, orig_coeffs, radii, scalers_dict)

        # Calculate noisey spherical harmonic coefficients, image with noise, and noise image
        true_noise = torch.randn_like(img_true)

        t = torch.randint(1, constants.timesteps + 1, (orig_coeffs.shape[0],)).to(
            constants.device
        )
        img_with_noise = perturb_input(img_true, t, true_noise)

        noise_level = t / constants.timesteps

        # Call model, loss, and backpropagation
        optimizer.zero_grad()

        img_pred_noise = model(
            img_with_noise, noise_level=noise_level, radio_flux=radio_flux
        )

        loss = F.mse_loss(true_noise, img_pred_noise)

        # Backpropagation
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameter
        optimizer.step()

        # Add loss to running loss
        epoch_loss += loss.item()
        offset = epoch * min(config.max_train_batches, len(train_dataloader))

        writer.add_scalar(
            "Training / Loss",
            loss.item(),
            offset + batch_idx,
        )

        writer.flush()

        # Update progress bar with current loss
        progress_bar.set_postfix(
            loss=loss.item(),
        )

        num_batches += 1

    # Average loss for the epoch
    epoch_loss /= num_batches

    # Log epoch loss to TensorBoard
    writer.add_scalar("Epoch / Training Loss", epoch_loss, epoch)

    print(f"Epoch [{epoch+1}/{config.epochs}], Training Loss: {epoch_loss:.4f}")


@torch.no_grad()
def do_test_loop(
    model,
    test_dataloader,
    scalers_dict,
    epoch,
    config,
    writer,
):
    model.eval()

    print("Starting testing epoch", epoch + 1)

    # Initialize progress bar
    progress_bar = tqdm(
        enumerate(test_dataloader),
        total=min(config.max_test_batches, len(test_dataloader)),
        desc=f"Epoch {epoch+1}/{config.epochs}",
    )

    # Testing: Loop through batches
    epoch_loss = 0.0
    radii = np.arange(
        config.min_radius, config.max_radius + config.res_radius, config.res_radius
    )
    num_batches = 0

    for batch_idx, (orig_coeffs, radio_flux) in progress_bar:
        # Break if reached max batch
        if batch_idx == config.max_train_batches:
            break

        # Move inputs to GPU
        orig_coeffs = orig_coeffs.to(constants.device)
        radio_flux = radio_flux.to(constants.device)

        # Normalize inputs
        img_true = get_potential_images(model, orig_coeffs, radii, scalers_dict)

        # Calculate noisey spherical harmonic coefficients, image with noise, and noise image
        true_noise = torch.randn_like(img_true)

        t = torch.randint(1, constants.timesteps + 1, (orig_coeffs.shape[0],)).to(
            constants.device
        )
        img_with_noise = perturb_input(img_true, t, true_noise)

        noise_level = t / constants.timesteps

        # Call model, loss, and backpropagation
        img_pred_noise = model(
            img_with_noise, noise_level=noise_level, radio_flux=radio_flux
        )

        # Backpropagation
        loss = F.mse_loss(true_noise, img_pred_noise)

        # Add loss to running loss
        epoch_loss += loss.item()

        # Update progress bar with current loss
        progress_bar.set_postfix(
            loss=loss.item(),
        )

        num_batches += 1

    epoch_loss /= num_batches

    # Log epoch loss to TensorBoard
    writer.add_scalar("Epoch / Testing Loss", epoch_loss, epoch)
    writer.flush()

    print(f"Epoch [{epoch+1}/{config.epochs}], Testing Loss: {epoch_loss:.4f}")


def get_potential_images(model, coeffs, radii, scalers_dict):
    B, nmax, _ = coeffs.shape
    nrad = radii.size

    images = torch.zeros((B, nrad, config.nlat, config.nlon))
    images = images.to(constants.device)

    for i in range(nrad):
        radial_scaling = torch.zeros(
            coeffs.shape, dtype=torch.float32, device=constants.device
        )

        for n in range(coeffs.shape[-1]):
            radial_scaling[:, n, :] = 1 / radii[i] ** (n + 1)

        img_chan = model.isht(radial_scaling * coeffs).float()
        img_chan = (
            torch.asinh(img_chan / scalers_dict["unscaled_abs"][i])
            / scalers_dict["std"][i]
        )
        images[:, i, :, :] = img_chan

    return images


def main():
    # Load dataloaders and scalers
    train_dataloader, test_dataloader = get_dataloaders(config)
    scalers_dict = get_scalers(config)

    # Setup base denoising diffusion model
    model = DiffusionModel().to(constants.device)

    # Setup the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{config.run_name}")

    # Epoch loop
    for epoch in range(config.epochs):
        do_train_loop(
            model,
            train_dataloader,
            scalers_dict,
            epoch,
            config,
            writer,
            optimizer,
        )

        do_test_loop(model, test_dataloader, scalers_dict, epoch, config, writer)

        save_checkpoint(model, config, epoch)

        # Do plotting every epoch
        if config.plot_br:
            do_br_plot(config, model, epoch, writer)
            
        if config.plot_field_lines:
            do_field_line_plot(config, model, epoch, writer)

        if config.plot_img:
            do_img_plot(config, model, epoch, writer)

        scheduler.step()

        writer.flush()

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
