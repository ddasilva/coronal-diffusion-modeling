# Coronal Diffusion Model

This repository holds code to train and sample a generative diffusion model of theg Solar Corona Magnetic Field. It is currently under review for the Journal of Geophysical Research: Machine Learning & Computation.
<img src="static/reverse_diffusion.gif">

## Pretrained weights

Pretrained weights, processed training data, scalers, and sampling support data are available on hugging face at [https://huggingface.co/ddasilva/coronal-diffusion-modeling](https://huggingface.co/ddasilva/coronal-diffusion-modeling).

## Running the Code

Code is included to train the model, as well as running the model to generate samples. Examples of sampling the model can be fonud in the `notebooks/` directory. To train the model, the `train.py` script is run with no arguments, with configuration found in `config.py`.

 To bootstrap the training process:

```
$ python prepare_dataset.py
$ python write_scalers.py
$ python write_spharm_fit_matrix.py
$ python train.py
```

If you want to skip `prepare_dataset.py`, download `test_dataset_gong_hemi_imag.h5` and `training_dataset_gong_hemi_imag.h5` from HuggingFace and confiure `test_dataset_path` and `train_dataset_path` in `config.py` to point to the paths.

If you want to skip `write_scalers.py`, download `scalers_gong.json` from HuggingFace and set `scalers_path` in `config.py` to the path.

If you want to skip `write_spharm_fit_matrix.py`, download `spharm_fit_mat_gong.h5` from HuggingFace and set `spharm_fit_mat_path` in `config.py` to the path.

If you want to skip to using sampling using one of the notebooks, do each of the above (except the requirements for `prepare_dataset.py` and download `experiment59-cs-fix-and-context-scaling_16.pth` from HuggingFace to the `checkpoints/` directory.