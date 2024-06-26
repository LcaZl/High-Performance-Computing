#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=0:02:00
#PBS -N imgconv
#PBS -q short_cpuQ
#PBS -o output/python/imgconv.out
#PBS -e output/python/imgconv.err

# Load python module
module load python-3.7.2

# Use previously created virtual environment with OpenCV (see local README)
source cv2/bin/activate

# Run python script (in virtual environment)
python HPC/dataset/generate_synthetic_images.py

# Deactivate the environment
deactivate