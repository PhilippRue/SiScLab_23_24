set -x


source /opt/nvidia_hpc_sdk/bin/enable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia_hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/

#found working link line with https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.19pcl4
nvfortran -openmp  -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl -liomp5 -lpthread -cudalib=cublas,nvtx,cusolver -acc -gpu=managed,lineinfo ../read_cmdline.f90 batched_zgemm.f90 -o test.exe


echo "linked libraries:"
ldd ./test.exe
