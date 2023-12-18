#!/bin/bash

### Project name
#SBATCH --account=lect0109

### Job name
#SBATCH --job-name=ZGEMM_test

### File for the output
#SBATCH --output=tests_OUTPUT

### Time your job needs to execute, e. g. 5 min
#SBATCH --time=00:05:00

### Use more than one node for parallel jobs on distributed-memory systems, e. g. 2
#SBATCH --nodes=1

### Number of CPUS per task (for distributed-memory parallelisation, use 1)
#SBATCH --cpus-per-task=16

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

### Number of processes per node, e. g. 6 (6 processes on 2 nodes = 12 processes in total)
#SBATCH --ntasks-per-node=1

### Use nodes with GPUs
#SBATCH --gres=gpu:1


# load compiler envoronment
module load NVHPC imkl/2023.1.0


# set the number of threads in your cluster environment to 1, as specified above
export OMP_NUM_THREADS=16

# nvtx profiling
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_LU_batched_GPU ./test.exe 
