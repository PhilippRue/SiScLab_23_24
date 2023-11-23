set -x
srun --account=lect0109 -n 1 -N 1 --ntasks-per-node=24 --gres=gpu:1 \
  nsys profile --trace=cuda,openacc,nvtx \
               --cuda-um-cpu-page-faults=true \
               --force-overwrite=true \
               -o out_test_LU ./test.exe 
