set -x
nvfortran -openmp -llapack -lblas -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo mod_matrix_tools.f90 mod_lu.f90 test_LU.f90 -o test.exe
