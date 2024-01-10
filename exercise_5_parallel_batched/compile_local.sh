# make intel-mkl available
source compiler-select intel-oneapi
# switch compiler environment to NVHPC
source /opt/nvidia_hpc_sdk/bin/enable

# compile
nvfortran -openmp -mp=multicore \
  -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl -liomp5 -lpthread \
  -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo \
  read_cmdline.f90 mod_matrix_tools.f90 mod_lu.f90 parallel_batched_lu_gpu.f90 -o test.exe

echo "done compiling"
