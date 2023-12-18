set -x

module load NVHPC
# switch to intelmkl underneath:
module load imkl/2023.1.0

nvfortran \
  -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl \
  -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo \
  ../read_cmdline.f90 mod_matrix_tools.f90 mod_lu.f90 batched_lu_gpu.f90 \
  -o test.exe


echo "linked libraries:"
ldd ./test.exe
