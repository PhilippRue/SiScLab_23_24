set -x
nvfortran -llapack -lblas -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo test_matmul.f90 -o test.exe
