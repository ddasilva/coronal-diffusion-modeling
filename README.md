# Coronal Diffusion Model

This repository holds code to train and sample a generative diffusion model of the Solar Corona Magnetic Field. The code is trained using ADAPT-WSA simulation potential field solutions from the [SuryaBench](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-coronal-extrapolation) dataset.

<img src="static/reverse_diffusion.gif">

## Pretrained weights

Pretrained weights are available on hugging face at [https://huggingface.co/ddasilva/coronal-diffusion-modeling](https://huggingface.co/ddasilva/coronal-diffusion-modeling).

## Publications

A publication preprint is available on arXiv at: [https://doi.org/10.48550/arXiv.2510.01441](https://doi.org/10.48550/arXiv.2510.01441)

## Running the Code

Code is included to train the model, as well as running the model to generate samples. Examples of sampling the model can be fonud in the `notebooks/` directory. To train the model, the `train.py` script is run with no arguments, with configuration found in `config.py`.

The full training process is:

```
$ python make_augmented_dataset.py
$ python write_scalers.py
$ python write_seed_helpers.py
$ python train.py
```
