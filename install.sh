#!/bin/bash
set -e  # exit immediately if a command fails

# Step 1: Create conda environment
echo ">>> Creating conda environment from environment.yml"
conda env create -f environment.yml

# Step 2: Activate environment
echo ">>> Activating environment gwtorch-env"
eval "$(conda shell.bash hook)"
conda activate gwtorch-env

# Step 2.5: Go back one directory
cd .. 

# Step 3: Clone teobresums (Dali branch)
if [ ! -d teobresums ]; then
    echo ">>> Cloning teobresums (Dali branch)"
    git clone -b Dali https://bitbucket.org/teobresums/teobresums.git
else
    echo ">>> teobresums already exists, skipping clone"
fi

# Step 4: Install Python bindings
echo ">>> Installing teobresums/Python"
cd teobresums/Python
pip install -e .

# Step 5: Install PyCBC bindings
echo ">>> Installing teobresums/PyCBC"
cd ../PyCBC
pip install -e .

# Step 6: Go back to project root
cd ../..

echo ">>> Setup complete! Environment 'gwtorch-env' is ready with teobresums installed."
