


set -x

# for benchmark
#for iomp in 1 2 4 8 16 32; do echo "#OpenMP: ${iomp}" && echo "#OpenMP: ${iomp}" >> timings.txt  && OMP_NUM_THREADS=${iomp} ./test.exe -n 1024 -m 10 | grep "timings batch strided" >> timings.txt; done

# nvtx profiling
export OMP_NUM_THREADS=16
nsys profile --trace=cuda,openacc,nvtx \
             --cuda-um-cpu-page-faults=true \
             --force-overwrite=true \
             -o out_test_ZGEMM_batched ./test.exe 
