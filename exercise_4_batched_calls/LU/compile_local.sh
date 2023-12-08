set -x

source /opt/nvidia_hpc_sdk/bin/enable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia_hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/

nvfortran -openmp \
  -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl -liomp5 -lpthread \
  -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo \
  ../read_cmdline.f90 mod_matrix_tools.f90 mod_lu.f90 batched_lu.f90 -o test.exe
