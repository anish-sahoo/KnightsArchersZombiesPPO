#!/bin/bash

# Create a Conda environment with Python 3.10
conda create --name mappo_env python=3.10 -y

# Activate the environment
conda activate mappo_env

# Install packages
conda install -y conda-forge::pettingzoo
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y conda-forge::matplotlib
conda install -y conda-forge::numpy
conda install -y conda-forge::tqdm

# Deactivate the environment
conda deactivate

echo "Environment mappo_env has been successfully created and configured."