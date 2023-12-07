set -x


source /opt/nvidia_hpc_sdk/bin/enable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia_hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/

nvfortran -llapack -lblas -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo batched_zgemm_gpu.f90 -o test.exe


echo "linked libraries:"
ldd ./test.exe
