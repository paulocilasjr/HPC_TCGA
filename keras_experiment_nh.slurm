#!/bin/bash
#SBATCH --job-name=keras_nh    # Job name
#SBATCH --output=keras_nh.out  # Output file
#SBATCH --error=keras_nh.err   # Error file
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=10             # Number of CPU cores
#SBATCH --mem=150G                     # Memory allocation
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=6-02:00:00              # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate keras_env

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install any required Python dependencies
pip install --upgrade --no-cache-dir tensorflow scikit-learn pandas

# Run the Keras script
python keras_experiment.py
