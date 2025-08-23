# Name of the run, used for model checkpoints and tensorboard logs
run_name = "experiment14-refactor-alldata"

# Training settings
epochs = 100
batch_size = 128

# Learning rate and scheduler
learning_rate = 0.0001
scheduler_step_size = 1
scheduler_gamma = 0.8

# Dataloader settings
train_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/training_dataset.h5"
test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset.h5"
#train_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/training_dataset_fast.h5"
#test_dataset_path = "/data/dedasilv/coronal-diffusion-modeling/test_dataset_fast.h5"

num_workers = 8

# Data Augmenter settings (used by make_augmented_dataset.py). Path to WSA
# FITS file directories and rotation delta
train_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
test_wsa_dir = "/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
delta_rot = 360

# Scalers and seed helpers
scalers_path = 'data/scalers.json'
seed_helper_max = 'data/seed_helper_max.json'
seed_helper_min = 'data/seed_helper_min.json'
#scalers_path = 'data/scalers_fast.json'
#seed_helper_max = 'data/seed_helper_max_fast.json'
#seed_helper_min = 'data/seed_helper_min_fast.json'

# Checkpoint plotting parameters for training. Flags whether to plot Br
# (magnetograms) and field lines (traces) after each epoch. When they
# are plotted, they are saved in tensorboard logs.
plot_br = False
plot_field_lines = True

# Loss Tradeoff Parameters Between Harmonis MSE and
# Magnetic Potential Loss Function Term (not used)
harmonics_lambda = 1
magnetic_lambda = 0

# Paths to reference images used by plot_reference_images.py
ref_min_path = '/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test/wsa_201010010800R000_ahmi.fits'
ref_max_path = '/data/dedasilv/coronal-diffusion-modeling/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test/wsa_201410110800R001_ahmi.fits'
