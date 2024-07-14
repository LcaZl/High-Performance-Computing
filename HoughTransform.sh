#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb -l place=pack:excl
#PBS -l walltime=0:50:00
#PBS -N ht
#PBS -q short_cpuQ
#PBS -o output/ht.out
#PBS -e output/ht.err

module load python-3.7.2
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load mpich-3.2.1--gcc-9.1.0

# Use previously created virtual environment with OpenCV (see README)
source cv2/bin/activate

# Dynamically set the environment variables from PBS directives
export PBS_SELECT=1 # Number of nodes
export PBS_NCPUS=$(qstat -f $PBS_JOBID | grep -oP '(?<=Resource_List.ncpus = )\d+') # (select * cpus) -> total number of cpus.
export PBS_MEM=$(qstat -f $PBS_JOBID | grep -oP '(?<=Resource_List.mem = )\d+' | sed 's/gb//') 
PARAM_FILE="HPC/parameters"
export NP_VALUE=$(grep "pbs_np=" $PARAM_FILE | cut -d '=' -f 2)
export OMP_PLACES=threads

mpiexec -np $NP_VALUE ./HPC/HoughTransform $PARAM_FILE
mpiexec -np $NP_VALUE ./HPC/HoughTransform $PARAM_FILE

# Unset the environment variables and deactivate virtual environment
unset PBS_SELECT
unset PBS_NCPUS
unset PBS_MEM
unset OMP_PLACES
unset NP_VALUE
deactivate

# MPIEXEC
# mpiexec --report-bindings -np 1 --map-by node:pe=2 --bind-to core  ./HPC/openMP/HoughTransform HPC/openMP/parameters
# or Use only -n <processes> for standard usage. For hybrid approach -np is indicated.
# "--map-by node:pe=2": This option means that each MPI process will span 2 cores.
# "--bind-to core": This option ensures that each process binds to the specific cores assigned to it.
# "-np": Run this many copies of the program on the given nodes.
# "--bind-to": Bind processes to the specified object, defaults to core. Supported options include slot, hwthread, core, l1cache, l2cache, l3cache, socket, numa, board, and none.
# "--report-bindings": Report any bindings for launched processes.
# "--map-by": Map  to the specified object, defaults to socket. Supported options include slot, hwthread, core, L1cache, L2cache, L3cache, socket, numa, board, node, sequential, distance, and ppr. 
# Any object can include modifiers by adding a ":" and:
# - any combination of PE=n (bind n processing elements to each proc), 
# - SPAN (load balance the processes across the  allocation),  
# - OVERSUBSCRIBE (allow  more  processes on a node than processing elements), and NOOVERSUBSCRIBE.  This includes PPR, where the pattern would be terminated by another colon to separate it from the modifiers.