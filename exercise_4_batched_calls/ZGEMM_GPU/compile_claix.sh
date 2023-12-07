set -x

module load NVHPC
nvfortran -llapack -lblas -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo batched_zgemm_gpu.f90 -o test.exe


echo "linked libraries:"
ldd ./test.exe
