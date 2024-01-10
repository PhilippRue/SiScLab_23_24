


set -x

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# nvtx profiling for a single OpenMP CPU thread
export OMP_NUM_THREADS=1
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_parallel_LU_batched_GPU_1 ./test.exe -o 20


# nvtx profiling for 4 OpenMP threads
export OMP_NUM_THREADS=4
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_parallel_LU_batched_GPU_4 ./test.exe -o 20
