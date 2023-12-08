


set -x

# nvtx profiling
export OMP_NUM_THREADS=32
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_LU_batched ./test.exe 
