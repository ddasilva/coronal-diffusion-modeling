# Name of the run, used for model checkpoints and tensorboard logs
run_name = "experiment35-more-data"

# Training settings
epochs = 100
batch_size = 20
#max_train_batches = 1
#max_test_batches = 1
max_train_batches = float("inf")
max_test_batches = float("inf")

# Learning rate and scheduler
learning_rate = 0.00001
scheduler_step_size = 100000
scheduler_gamma = 0.8

# Spherical Harmonic Settings
nmax = 90
X_SIZE = 8281

# Dataloader settings
# train_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/training_dataset_gong.h5"
# test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset_gong.h5"
train_dataset_path = (
    "/data/dedasilv/coronal-diffusion-modeling/training_dataset_gong.h5"
)
test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset_gong.h5"

num_workers = 16

# Data Augmenter settings (used by make_augmented_dataset.py). Path to WSA
# FITS file directories and rotation delta

train_wsa_dir = '/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation_GONG/train'
test_wsa_dir = '/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation_GONG/test'
#train_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
#test_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"

delta_rot = 360  # degrees

# Scalers and seed helpers
scalers_path = f"data/scalers_gong.json"

# Checkpoint plotting parameters for training. Flags whether to plot Br
# (magnetograms) and field lines (traces) after each epoch. When they
# are plotted, they are saved in tensorboard logs.
plot_br = True
plot_field_lines = True
plot_img = True
