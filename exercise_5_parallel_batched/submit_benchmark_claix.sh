#!/bin/bash

### Project name
#SBATCH --account=lect0109

### Job name
#SBATCH --job-name=ZGEMM_test

### File for the output
#SBATCH --output=tests_OUTPUT

### Time your job needs to execute, e. g. 5 min
#SBATCH --time=02:30:00

### Use more than one node for parallel jobs on distributed-memory systems, e. g. 2
#SBATCH --nodes=1

### Number of CPUS per task (for distributed-memory parallelisation, use 1)
#SBATCH --cpus-per-task=8

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

### Number of processes per node, e. g. 6 (6 processes on 2 nodes = 12 processes in total)
#SBATCH --ntasks-per-node=1

### Use nodes with GPUs
#SBATCH --gres=gpu:2

### take node exclusively
#SBATCH --exclusive

### all the memory of the node
#SBATCH --mem=0


# load compiler envoronment
module load NVHPC imkl/2023.1.0
module load GCC/12.3.0 SciPy-bundle/2023.07

echo "python version"
python --version

echo "start benchmark"
python run_benchmark.py
#python test_python.py 
echo "end benchmark"
