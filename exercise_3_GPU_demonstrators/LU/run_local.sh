set -x
export OMP_NUM_THREADS=24
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_LU ./test.exe 
