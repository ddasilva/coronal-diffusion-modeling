import numpy as np

# Name of the run, used for model checkpoints and tensorboard logs
run_name = "experiment44-sliding-radii"

# Training settings
epochs = 100
batch_size = 32
max_train_batches = float("inf")
max_test_batches = float("inf")
# max_train_batches = 1
# max_test_batches = 1

restart_file = "checkpoints/experiment44-sliding-radii_7.pth"  # preload these weights prior to training
start_epoch = 7  # zero indexed

# Learning rate and scheduler
learning_rate = 0.0001
scheduler_step_size = 100000
scheduler_gamma = 0.8

# Spherical Harmonic Settings
nmax = 90
fit_nmax = 45
X_SIZE = 8281

# Dataloader settings
train_dataset_path = (
    "/data/dedasilv/coronal-diffusion-modeling/training_dataset_noazrot.h5"
)
test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset_noazrot.h5"
num_workers = 16

# Data Augmenter settings (used by make_augmented_dataset.py). Path to WSA
# FITS file directories and rotation delta
train_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
test_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
delta_rot = 360  # degrees (set to no rotation- not needed for CNN)

# Scalers and spherical harmonic fix matrix
scalers_path = "data/scalers_16log.json"
spharm_fit_mat_path = "data/spharm_fit_mat_16log.h5"

# Checkpoint plotting parameters for training. Flags whether to plot Br
# (magnetograms) and field lines (traces) after each epoch. When they
# are plotted, they are saved in tensorboard logs.
plot_br = True
plot_field_lines = True
plot_img = True

# 3D settings
min_radius = 1.0
max_radius = 2.5
radii = np.logspace(np.log10(min_radius), np.log10(max_radius), 16)

nlat = 90
nlon = 180
