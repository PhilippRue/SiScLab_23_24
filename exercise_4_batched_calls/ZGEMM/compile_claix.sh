set -x

module load NVHPC
#nvfortran -llapack -lblas -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo batched_zgemm.f90 -o test.exe

# built-in blas does not have batched zgemm:
# /rwthfs/rz/cluster/home/pr357554/SiScLab_23_24/exercise_4_batched_calls/ZGEMM/batched_zgemm.f90:122: undefined reference to `zgemm_batch_strided_'
# switch to intelmkl underneath:
module load imkl/2023.1.0

nvfortran \
  -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl \
  -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo \
  ../read_cmdline.f90 batched_zgemm.f90 \
  -o test.exe




echo "linked libraries:"
ldd ./test.exe
