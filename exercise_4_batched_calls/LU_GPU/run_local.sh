


set -x

# nvtx profiling
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_LU_batched_GPU ./test.exe 
