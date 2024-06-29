#!/bin/bash
#PBS -l select=2:ncpus=6:mem=8gb -l place=scatter:excl
#PBS -l walltime=0:50:00
#PBS -N ja_tests_hts
#PBS -q short_cpuQ
#PBS -o output/tests/ht_6_2_scatter:excl.out
#PBS -e output/tests/ht_6_2_scatter:excl.err

module load python-3.7.2
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load mpich-3.2.1--gcc-9.1.0

# Use previously created virtual environment with OpenCV (see README)
source cv2/bin/activate
PARAM_DIR="HPC/tests/t_6_2_scatter:excl"

for PARAM_FILE in $PARAM_DIR/parameters_*; do

    # Dynamically set the environment variables from PBS directives
    export PBS_SELECT=$(cat $PBS_NODEFILE | sort | uniq | wc -l) # Number of nodes
    export PBS_NCPUS=$(qstat -f $PBS_JOBID | grep -oP '(?<=Resource_List.ncpus = )\d+') # (select * cpus) -> total number of cpus.
    export PBS_MEM=$(qstat -f $PBS_JOBID | grep -oP '(?<=Resource_List.mem = )\d+' | sed 's/gb//') 
    export NP_VALUE=$(grep "pbs_np=" "$PARAM_FILE" | cut -d '=' -f 2)
    export OMP_PLACES=threads

    # Print the environment variables
    echo "Running with PARAM_FILE=$PARAM_FILE and NP_VALUE=$NP_VALUE"
    echo "PBS_SELECT=$PBS_SELECT"
    echo "PBS_NCPUS=$PBS_NCPUS"
    echo "PBS_MEM=$PBS_MEM"
    echo "OMP_PLACES=$OMP_PLACES"
    echo "NP_VALUE=$NP_VALUE"

    # Get the parameter file for this array job
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    
done

# Unset the environment variables and deactivate virtual environment
unset PBS_SELECT
unset PBS_NCPUS
unset PBS_MEM
unset OMP_PLACES
unset NP_VALUE
deactivate
