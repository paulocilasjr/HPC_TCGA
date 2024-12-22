#!/bin/bash
#SBATCH --job-name=now_resnet_nh     # Job name
#SBATCH --output=now_resnet_nh.out  # Output file
#SBATCH --error=now_resnet_nh.err   # Error file
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=10              # Number of CPU cores
#SBATCH --mem=150G                      # Memory allocation
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=6-02:00:00               # Time limit (days-HH:MM:SS)

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate ludwig

# Ensure the correct library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATHy

# Reinstall bitsandbytes if necessary
# pip install bitsandbytes --upgrade --no-cache-dir

# Run the Ludwig experiment
ludwig experiment --dataset /share/lab_goecks/TCGA_deep_learning/now_nh_experiment/er_status_no_white.csv --config /share/lab_goecks/TCGA_deep_learning/config_files/resnet_nh_config.yml --output_directory now_resnet_nh
