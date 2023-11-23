! test program for GPU-accelerated matrix-matrix multiplication
program test

  use cublas_v2
  use openacc
  use nvtx

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: n = 1000, nmax = 10
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  real (kind=dp), allocatable :: tmp(:, :), tmp2(:,:)
  complex (kind=dp), allocatable :: a(:,:), b(:,:), c(:,:), c2(:,:)
  integer :: ierr
  integer :: clock_rate, start_time, stop_time
  real (kind=dp) :: timing1, timing2, timing3
  character (100) :: fmt
  type (cublasHandle) :: handle_cublas

  ! create cublas handle
  ierr = cublasCreate(handle_cublas)
  ierr = ierr + cublasSetStream(handle_cublas, acc_get_cuda_stream(acc_async_sync))
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasCreate'

  ! allocate memory of working arrays
  allocate(a(n,n), b(n,n), c(n,n), c2(n,n))
  allocate(tmp(n,n), tmp2(n,n))

  ! find clock rate for timing measurements
  call system_clock(count_rate=clock_rate) ! Find the rate


  ! initialize A matrix with random numbers + unity matrix
  a = czero
  do i = 1, n
    a(i, i) = cone
  end do
  call random_number(tmp)
  call random_number(tmp2)
  a = a + cmplx(tmp, tmp2)

  ! initialize B matrix with random numbers
  b = cone
  call random_number(tmp)
  call random_number(tmp2)
  b = b + cmplx(tmp, tmp2)


  ! start a measurement reagion in nvtx
  call nvtxStartRange('ZGEMM - GPU', 1)

  ! start timing measurement
  call system_clock(count=start_time)

  ! matrix-matrix multiply
  ! use Zgemm on GPU from cublas: https://docs.nvidia.com/hpc-sdk/compilers/fortran-cuda-interfaces/index.html#dc-zgemm
  !$acc data create(c)
  !$acc host_data use_device(a, b, c)
  ierr = cublasZgemm_v2(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, cone, a, n, b, n, czero, c, n)
  if (ierr /= CUBLAS_STATUS_SUCCESS) stop 'Error in cublasZgemm_v2 of matmat'
  !$acc end host_data
  !$acc end data
  !$acc wait

  call system_clock(count=stop_time) ! Stop timing
  timing1 = (stop_time-start_time)/real(clock_rate)

  ! end a measurement reagion in nvtx
  call nvtxEndRange()

  ! print output
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# A matrix'
  do i = 1, nmax
    write(*,fmt) a(i, 1:nmax)
  end do
  write(*,'(A)') '# B matrix'
  do i = 1, nmax
    write(*,fmt) b(i, 1:nmax)
  end do
  write(*,'(A)') '# C matrix: C = A * B'
  do i = 1, nmax
    write(*,fmt) c(i, 1:nmax)
  end do
  write(*, *) 
  write(*, *) 'timings:', timing1

  ! start a measurement reagion in nvtx
  call nvtxStartRange('ZGEMM - BLAS', 2) ! icolor argument is different for better visibility

  call system_clock(count=start_time)

  ! for validation compare with ZGEMM on CPU from LAPACK
  call zgemm('N', 'N', n, n, n, (1.0_dp, 0.0_dp), a, n, b, n, (0.0_dp, 0.0_dp), c2, n)

  call system_clock(count=stop_time) ! Stop timing
  timing2 = (stop_time-start_time)/real(clock_rate)

  ! end last measurement region in nvtx
  call nvtxEndRange()

  ! write out alternative result and compare with GPU version
  write(*, *)
  write(*,'(A)') '# C matrix BLAS on CPU'
  do i = 1, nmax
    write(*,fmt) c2(i, 1:nmax)
  end do
  write(*,'(A)') '# Difference to GPU'
  do i = 1, nmax
    write(*,fmt) c(i, 1:nmax) - c2(i, 1:nmax)
  end do
  write(*, *) 'max difference', maxval(real(c-c2, kind=dp))

  write(*, *) 
  write(*, *) 'timings:', timing1, timing2


  ! start a measurement reagion in nvtx
  call nvtxStartRange('MATMUL', 3) ! icolor argument is different for better visibility

  call system_clock(count=start_time)

  ! for validation compare with matmul instead of LAPACK
  c2 = matmul(a, b)

  call system_clock(count=stop_time) ! Stop timing
  timing3 = (stop_time-start_time)/real(clock_rate)

  ! end last measurement region in nvtx
  call nvtxEndRange()

  ! write out alternative result and compare with GPU version
  write(*, *)
  write(*,'(A)') '# C matrix from matmul instead of LAPACK'
  do i = 1, nmax
    write(*,fmt) c2(i, 1:nmax)
  end do
  write(*,'(A)') '# Difference to GPU result'
  do i = 1, nmax
    write(*,fmt) c(i, 1:nmax) - c2(i, 1:nmax)
  end do
  write(*, *) 'max difference', maxval(real(c-c2, kind=dp))

  ! collect timings
  write(*, *) 
  write(*, *) 'timings:', timing1, timing2, timing3
  write(*, *) 'speedup LAPACK vs matmul', timing3/timing2
  write(*, *) 'speedup GPU vs LAPACK', timing2/timing1
  write(*, *) 'speedup GPU vs matmul', timing3/timing1

  ! clean up memory allocations
  deallocate(a, b, c, c2)

  ! destroy cublas handle
  ierr = cublasDestroy(handle_cublas)
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasDestroy'

end program test
