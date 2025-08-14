#!/usr/bin/env bash
set -e  # Stop on error

# ===============================
# 1. Create and activate Conda env
# ===============================
ENV_NAME="gwtorch-env"
PYTHON_VERSION="3.11"

echo ">>> Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# ===============================
# 2. Install compilers & math libs
# ===============================
echo ">>> Installing GCC/G++ and math libraries"
conda install -y gcc_linux-64 gxx_linux-64
conda install -y -c conda-forge gmpy2 mpmath

# ===============================
# 3. Install gwmat (pinned commit)
# ===============================
echo ">>> Cloning and installing gwmat"
GW_MAT_COMMIT="master"  # Change to a commit hash or tag for reproducibility
git clone https://git.ligo.org/anuj.mishra/gwmat.git
cd gwmat
#git checkout $GW_MAT_COMMIT
pip install .
cd ..

# ===============================
# 4. Install gwtorch with 'all' extras
# ===============================
echo ">>> Installing gwtorch dependencies"
# Install torch/torchvision with CUDA 12.8 wheels
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install your package
pip install -e .[all]

# ===============================
# 5. Lock environment (optional)
# ===============================
echo ">>> Saving environment to requirements.lock.txt"
pip freeze > requirements.lock.txt

echo ">>> Installation complete!"
