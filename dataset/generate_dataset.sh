#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=0:02:00
#PBS -N imgconv
#PBS -q short_cpuQ
#PBS -o imgconv.out
#PBS -e imgconv.err

# Load python module
module load python-3.7.2

# Use previously created virtual environment with OpenCV (see local README)
source cv2/bin/activate

# Run python script (in virtual environment)
python HPC/openMP/dataset/generate_dataset.py

# Deactivate the environment
deactivate