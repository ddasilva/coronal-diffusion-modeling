# Name of the run, used for model checkpoints and tensorboard logs
run_name = "experiment15-16harmonics"

# Training settings
epochs = 100
batch_size = 128
X_SIZE = 8281
nmax = 90

# Learning rate and scheduler
learning_rate = 0.0001
scheduler_step_size = 1
scheduler_gamma = 0.8

# Dataloader settings
train_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/training_dataset.h5"
test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset.h5"

num_workers = 16

# Data Augmenter settings (used by make_augmented_dataset.py). Path to WSA
# FITS file directories and rotation delta
train_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
test_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
delta_rot = 1   # degrees

# Scalers and seed helpers
scalers_path = f'data/scalers.json'
seed_helper_max = f'data/seed_helper_max.json'
seed_helper_min = f'data/seed_helper_min.json'

# Checkpoint plotting parameters for training. Flags whether to plot Br
# (magnetograms) and field lines (traces) after each epoch. When they
# are plotted, they are saved in tensorboard logs.
plot_br = True
plot_field_lines = True

# Loss Tradeoff Parameters Between Harmonis MSE and Magnetic Potential Loss
# Function Term (not used for NeurIPS paper)
harmonics_lambda = 1
magnetic_lambda = 0
