#!/bin/bash
#SBATCH --job-name=pytorch_exp     # Job name
#SBATCH --output=pytorch_exp.out  # Output file
#SBATCH --error=pytorch_exp.err   # Error file
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=10        # Number of CPU cores
#SBATCH --mem=500G                # Memory allocation
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=6-02:00:00         # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_env  # Replace 'pytorch_env' with your Conda environment name

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the PyTorch script
python pytorch_experiment.py
