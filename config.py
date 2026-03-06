import numpy as np

# Name of the run, used for model checkpoints and tensorboard logs
run_name = "experiment54-hemi"

# Training settings
epochs = 500
batch_size = 12
max_train_batches = float("inf")
max_test_batches = float("inf")
#max_train_batches = 1
#max_test_batches = 1

restart_file = None  # preload these weights prior to training
start_epoch = 0  # zero indexed

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
    "/data/dedasilv/coronal-diffusion-modeling/training_dataset_gong_hemi.h5"
)
test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset_gong_hemi.h5"
num_workers = 16

# Data Augmenter settings (used by make_augmented_dataset.py). Path to WSA
# FITS file directories and rotation delta
train_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation_GONG/train"
test_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation_GONG/test"

# Scalers and spherical harmonic fix matrix
scalers_path = "data/scalers_gong.json"
spharm_fit_mat_path = "data/spharm_fit_mat_gong.h5"

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
