# SiSc Lab Project "Solving the Scattering Problem of Electrons at Atoms on the GPU"

## Background

The scattering or electrons in a solid at an atom is described by a well-known radial Green function depending on the angular momentum. The Green function is a product of the regular and irregular solutions on a not equally spaced radial grid of a Sturm Liouville type differential equation for each angular momentum, known as the Schrödinger or Dirac equation. To improve the accuracy, the regular and irregular solution are expanded in Chebychev functions. In practical simulation codes this scattering problem hast to be solved millions of times. Therefore we believe that bringing this problem on GPUs would be very beneficial. This would be the main task of the project.

- [JuKKR DFT code](https://jukkr.fz-juelich.de)
- [JuKKR git repo](https://iffgit.fz-juelich.de/kkr/jukkr)

### Radial Schrödinger equation

In density functional theory the ground state of the electronic structure in a material is found from solving the Schrödinger equation
$$[-\nabla^2 + V(\vec{r})]\Psi(\vec{r}; E) = E\Psi(\vec{r}; E),$$
where $V(\vec{r})$ is the potential coming from the positively charge nuclei and $\Psi(\vec{r}; E)$ is the wavefunction of an electon whose energy is $E$ and $\vec{r}$ is the spatial coordinate.

In the expansion of the wavefunction around the atoms it is beneficial to use spherical coordinates
$$\Psi(\vec{r}; E) = \sum_{l=1}^{l_{max}}\sum_{m=-l}^{l}R_l(r; E)Y_l^m(\hat{r})$$
where $Y_l^m(\hat{r})$ are spherical harmonics and $R_l(r)$ are the radial functions for which the radial Schrödinger equation 
$$\left[-\frac{1}{r}\frac{\partial^2}{\partial r^2}r + \frac{l(l+1)}{r^2} + V(r) - E\right] R_l(r;E) = 0$$
holds.

For more details see equation discussion in [Mavropoulos and Papanikolaou](https://juser.fz-juelich.de/record/50027/files/FZJ-2014-02214.pdf) [equation (53) gives the radial Schrödinger equation].

## Compiling a GPU application on CLAIX

- [CLAIX overview](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/fbd107191cf14c4b8307f44f545cf68a/)
- [CLAIX access](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/14573fc745ee478ba855539c240108b6/)
- [Available GPU Hardware on CLAIX](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/3fb4cb953142422dbbb656c1c3253cff/)

```bash
# connect to a CLAIX2018 login node which is equipped with GPUs
$ ssh YOUR-USER-ID@login18-g-1.hpc.itc.rwth-aachen.de

# load NVIDIA fortran compiler environment
$ module load NVHPC
  
# check availability of GPUs on this cluster node
bash-4.4$ nvaccelinfo 

CUDA Driver Version:           12000
NVRM version:                  NVIDIA UNIX x86_64 Kernel Module  525.125.06  Tue May 30 05:11:37 UTC 2023

Device Number:                 0
Device Name:                   Tesla V100-SXM2-16GB
Device Revision Number:        7.0
Global Memory Size:            16935419904
Number of Multiprocessors:     80
Concurrent Copy and Execution: Yes
...

# compiler for an application
$ nvfortran --version

nvfortran 21.11-0 64-bit target on x86-64 Linux -tp skylake-avx512 
NVIDIA Compilers and Tools
Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
```


### Excercise
- Go through [OpenACC introduction and tutorial](https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/openacc/basics/)
    --> try to do the hello world examples in FORTRAN on CLAIX

<details>
<summary>Solution</summary>
    
See code in `exercise_1_compile_a_simple_example/` of the [github repository](https://github.com/PhilippRue/SiScLab_23_24).

Compiling and test run:
```bash
# compile code without acceleration (`-Minfo=all` gives us information about parallelization)
$ nvfortran -fast -Minfo=all -acc=gpu hello_world.f90 -o test.exe
print_hello_world:
      4, Loop not vectorized/parallelized: contains call

# code with OpenACC directives
$ nvfortran -fast -Minfo=all -acc=gpu hello_world_openACC.f90 -o test_openacc.exe
print_hello_world:
      5, Accelerator serial kernel generated
         Generating NVIDIA GPU code
          5, !$acc loop seq

$ ./test.exe 
 hello world
 hello world
 hello world
 hello world
 hello world
 
 $ ./test_openacc.exe 
 hello world
 hello world
 hello world
 hello world
 hello world
```
</details>

### Using the `lect0109` project on CLAIX

For this course we have applied for computing time on the GPU partition of CLAIX. Under the project `lect0109` 0.048 Million core-h are available.
```
$ r_wlm_usage -q -p lect0109
Account:                                  lect0109
Type:                                         lect
Start of Accounting Period:             01.11.2023
End of Accounting Period:               30.04.2024
State of project:                           active
--------------------------------------------------
Quota monthly (core-h):                       8072
Remaining  core-h of prev. month:                0
Consumed   core-h current month:                 0
Consumed   core-h last 4 weeks:                  0
Consumable core-h (%):                         200
Consumable core-h:                           16143
--------------------------------------------------
Total quota (core-h):                    0.048 Mio
Total consumed core-h so far:            0.000 Mio
--------------------------------------------------
Default partition:                            c18m
Allowed partitions:                      c18m,c18g
Max. allowed wallclocktime:              4.0 hours
Max. allowed cores per job:                     48
```

You have all been added to the project and you can now run calculation using this project.
- [Slurm commands to use GPU parition](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/#Submitting%20a%20GPU%20job)
-  Example running a job on a GPU node (use a single GPU and a single CPU process):
```
srun --account=lect0109 -n 1 -N 1 --ntasks-per-node=1 --gres=gpu:1 ./test.exe
```


## Implementation of the radial solver in JuKKR

The implementation of the radial solver in the JuKKR code is explained in detail in Chapter 5 of the [PhD thesis of David Bauer](https://publications.rwth-aachen.de/record/229375).

The idea is to solve the [Lippmann-Schwinger equation](https://en.wikipedia.org/wiki/Lippmann–Schwinger_equation) that connects the incoming to the scattered wave in a scattering problem. Numerically this is done using Chebychev polynomials for an efficient and highly accurate algorithm. This allows to transform the system of differential equations of the radial Schrödinger equation to an integral equation. In the Chebychev basis this leads to the following form of the equations (see equation 5.70 of Bauers PhD thesis)
$$\underline{\underline{A}} \ \underline{\underline{U}} = \underline{\underline{J}}$$
where $\underline{\underline{A}}$, $\underline{\underline{U}}$ and $\underline{\underline{J}}$ are complex-valued matrices of double precision. Thus a system of linear equations has to be solved which is done using [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition). The CPU version of the algorithm employs the [`ZGETRF`](https://netlib.org/lapack/explore-html/d3/d01/group__complex16_g_ecomputational_ga5b625680e6251feb29e386193914981c.html), [`ZGETRS`](https://netlib.org/lapack/explore-html/d3/d01/group__complex16_g_ecomputational_ga3a5b88a7e8bf70591e521e86464e109d.html) and [`ZGEMM`](https://www.netlib.org/lapack/explore-html/dc/d17/group__complex16__blas__level3_ga4ef748ade85e685b8b2241a7c56dd21c.html) functions of the LAPACK library to perform the necessary LU decomposition (factorization, then solve) and matrix-matrix multiplications, respectively.

**Excercise:**

Write a minimal FORTRAN program that does an LU decomposition and a matrix matrix multiplication with double complex matrices using  the LAPACK library:
- initialize double complex matrices in FORTRAN
- minimal demonstrator for matrix matrix multiplication with LAPACK
- write another minimal demonstrator for an LU decomposition with LAPACK
- check correctness
- Hints:
    * link BLAS and LAPACK with the `nvfortran` compiler
    ```
    nvfortran -llapack -lblas test.f90 -o test.exe
    ```
    * `matmul` for built-in matrix-matrix multiplication

<details>
<summary>Solution</summary>

See code in `exercise_2_LAPACK_demonstrators/` of the [github repository](https://github.com/PhilippRue/SiScLab_23_24).

</details>


## Minimal demonstrator code for GPU acceleration

Before going through the actual implementation of the radial solver, we need to familiarize ourselves with the use of GPU-accelerated libraries and available tools to measure performance. For this we use an example program that only does an LU decomposition for a given matrix where we know the outcome already (e.g. you can check the result with [this online tool](https://www.emathhelp.net/en/calculators/linear-algebra/lu-decomposition-calculator/)).

**Excercise:**

Change the minimal example for LU decomposition and matrix-matrix multiplication to use cuda-accelerated libraries.

- GPU version of `ZGEMM` is part of the [cublas library](https://docs.nvidia.com/cuda/cublas/index.html?highlight=gemm#cublas-t-gemm)
- GPU version of `ZGETRF/S` is in the [cusolver library](https://docs.nvidia.com/hpc-sdk/compilers/fortran-cuda-interfaces/index.html#cfsolver-legacy-zgetrf) 
- Tools to understand/measure how GPU parallelization works and if it actually uses GPU hardware: [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems), download it [here](https://developer.nvidia.com/gameworksdownload).
- Some hints:
    * use `-cudalib=cublas,cusolver,nvtx` compiler flags for cublas, cusolver and nvtx library (the last one is needed to create traces for the Nsight Systems tool)
    * use `-gpu=managed` to allow compiler to manage the memory handling between CPU and GPU, in this way we only need a few OpenACC pragmas to inform the compiler about data which is handled on the CPU vs on the GPU. The data transfer etc is then done by the compiler.
    * also use `-acc` to activate OpenACC pragmas
    * cublas and cusolver need (different) handles to set their respective streams (here cublas code snippet): 
    ```
    use cublas_v2 [, only: cublasCreate, ...]
    use acc
    
    ierr = cublasCreate(handle_cublas)
    ierr = ierr + cublasSetStream(handle_cublas, acc_get_cuda_stream(acc_async_sync))
    if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasCreate'
    ```
    * use OpenACC data pragmas to help compiler determine data locality
    ```
    !$acc data create(matout) ! for data created on the GPU
    !$acc host_data use_device(mat1, mat2, matout) ! tell compiler that copy between CPU and GPU is needed for this data
    ! then use cublas ZGEMM call to multiply mat1 and mat2 which creates matout
    ...
    ! finally end the OpenACC data environment and add a wait clause to make sure data is synchronized
    !$acc end host_data
    !$acc end data
    !$acc wait
    ```
    * GPU correctness checking is possible with `compute-sanitizer` tool from NVHPC:
    ```bash
    # after compilation run
    $ compute-sanitizer ./test.exe
    ========= COMPUTE-SANITIZER
    ...
    ========= ERROR SUMMARY: 0 errors
    ```
    * tracing with Nsight Systems:
    ```
    $ nsys profile --trace=cuda,openacc,nvtx --cuda-um-cpu-page-faults=true -o out_test ./test.exe
    ...
    Generating '.../login18-g-1_171321/nsys-report-6cf5.qdstrm
    [1/1] [========================100%] out_test.nsys-rep
    Generated:
    .../exercise_3_GPU_demonstrators/out_test.nsys-rep
    ```

<details>
<summary>Solution</summary>

See code in `exercise_3_GPU_demonstrators` of the [github repository](https://github.com/PhilippRue/SiScLab_23_24).

</details>

## Benchmarking and tuning of the implementation

- use OpenMP threaded BLAS/LAPACK library for parallel execution on CPU
- measure performance for different matrix sizes
- use OpenACC directives to merge data regions and fuze two matrix-matrix multiplications together
- batched version of ZGEMM & LU decomposition

## Mini-app for radial Schrödinger equation in JuKKR code

Will come next ...
