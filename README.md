# Denoising Diffusion Probabilistic Model (DDPM) Implementation

This repository contains an implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch, designed for generating high-quality 128x128 RGB images.

## Overview

DDPMs are a class of generative models that learn to generate images by gradually denoising random Gaussian noise. The model is trained to reverse a forward diffusion process that slowly adds noise to images according to a fixed schedule.

Key components:

- Custom U-Net architecture with time embeddings for noise prediction
- Forward diffusion process with controlled noise schedule
- Training loop with diffusion loss function
- Custom dataset loader for butterfly images

## Model Architecture

The implementation uses a U-Net with:
- 3 encoder blocks with increasing channels (64->128->256)
- 2 decoder blocks with skip connections
- Time embeddings injected at each block
- Batch normalization and ReLU activations
- Optimized for 128x128 RGB images

## Training Process

The model is trained by:
1. Adding noise to images according to randomly sampled timesteps
2. Predicting the noise using the U-Net
3. Calculating MSE loss between predicted and actual noise
4. Updating model parameters via backpropagation

Training hyperparameters:
- Learning rate: 1e-4
- Batch size: 2
- Number of epochs: 100
- Noise schedule: β from 0.0001 to 0.02

## Requirements

- PyTorch
- torchvision
- PIL
- numpy
- matplotlib

## Usage

1. Prepare your image dataset in a directory
2. Update the data_dir path in the notebook
3. Run the training loop
4. Model checkpoints will be saved after each epoch

## Theory

The implementation is based on the DDPM framework where:
- Forward process gradually adds Gaussian noise
- Reverse process learns to denoise through a neural network
- Training maximizes the evidence lower bound (ELBO)
- Loss function focuses on noise prediction at each timestep

## License

This project is available under the MIT License.
