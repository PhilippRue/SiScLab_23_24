


set -x

# nvtx profiling
srun --account=lect0109 -n 1 -N 1 --ntasks-per-node=16 --gres=gpu:1 \
  OMP_NUM_THREADS=16 \
  nsys profile --trace=cuda,openacc,nvtx \
               --cuda-um-cpu-page-faults=true \
               --force-overwrite=true \
               -o out_test_ZGEMM_batched_GPU ./test.exe 
