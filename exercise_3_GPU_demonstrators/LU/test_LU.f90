! test program for GPU-accelerated matrix-matrix multiplication
program test

  use cublas_v2
  use cusolverDn
  use openacc
  use nvtx
  use mod_lu, only: linearsolve_dc, linearsolve_dc_GPU
  use mod_matrix_tools, only: init_matrices
  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: n = 10, m = 2
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  complex (kind=dp), allocatable :: a(:,:), a2(:,:), b(:,:), b2(:,:), c(:,:)
  character(100) :: fmt

  integer :: ierr
  integer :: clock_rate, start_time, stop_time
  real (kind=dp), allocatable :: timings(:)
  type (cublasHandle) :: handle_cublas
  type (cusolverDnHandle) :: handle_cusolver


  ! create cublas handle for ZGEMM and cusolver handle for LU decomposition
  ierr = cublasCreate(handle_cublas)
  ierr = ierr + cublasSetStream(handle_cublas, acc_get_cuda_stream(acc_async_sync))
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasCreate'

  ierr = cusolverDnCreate(handle_cusolver)
  ierr = ierr + cusolverDnSetStream(handle_cusolver, acc_get_cuda_stream(acc_async_sync))
  if (ierr/=0) stop 'Error in cusovlerCreate'

  ! find clock rate for timing measurements
  call system_clock(count_rate=clock_rate) ! Find the rate


  ! ================== CPU version ==================

  ! allocate memory and initialize arrays
  call init_matrices(n, m, a, a2, b, b2, c, timings)

  ! start a measurement region in nvtx
  call system_clock(count=start_time)
  call nvtxStartRange('LU - CPU', 1) ! icolor argument is different for better visibility

  ! solve linear system of equations using LU decomposition
  ! b is overwritten in here
  call linearsolve_dc(a, b)

  ! end a measurement reagion in nvtx and measure time
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(1) = (stop_time-start_time)/real(clock_rate)


  ! start region for ZGEMM
  call system_clock(count=start_time)
  call nvtxStartRange('ZGEMM - CPU', 2) ! icolor argument is different for better visibility

  ! check result with a zgemm (we solve a*b=c which we can check here)
  !c = matmul(a2, b)
  call zgemm('N', 'N', n, m, n, cone, a2, n, b, n, czero, c, n)

  ! end region for ZGEMM
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(2) = (stop_time-start_time)/real(clock_rate)


  ! check c == b2
  write(*, *)
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# result after zgemm'
  do i = 1, n
    write(*,fmt) c(i, :)
  end do
  write(*,'(A)') '# Difference to input'
  do i = 1, n
    write(*,fmt) c(i, :) - b2(i, :)
  end do
  write(*, *) 'max difference', maxval(real(c-b2, kind=dp))


  ! ================== GPU version ==================

  ! allocate memory and initialize arrays
  call init_matrices(n, m, a, a2, b, b2, c, timings)

  ! start a measurement region in nvtx
  call system_clock(count=start_time)
  call nvtxStartRange('LU - GPU', 3) ! icolor argument is different for better visibility

  ! solve linear system of equations using LU decomposition
  ! b is overwritten in here
  call linearsolve_dc_GPU(a, b, handle_cusolver)

  ! end a measurement reagion in nvtx and measure time
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(3) = (stop_time-start_time)/real(clock_rate)


  ! start region for ZGEMM
  call system_clock(count=start_time)
  call nvtxStartRange('ZGEMM - GPU', 4) ! icolor argument is different for better visibility

  ! check result with a zgemm (we solve a*b=c which we can check here)  
  !$acc data create(c)
  !$acc host_data use_device(a2, b, c)
  ierr = cublasZgemm_v2(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, cone, a2, n, b, n, czero, c, n)
  if (ierr /= CUBLAS_STATUS_SUCCESS) stop 'Error in cublasZgemm_v2 of matmat'
  !$acc end host_data
  !$acc end data
  !$acc wait

  ! end region for ZGEMM
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(4) = (stop_time-start_time)/real(clock_rate)


  ! check c == b2
  write(*, *)
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# result after zgemm'
  do i = 1, n
    write(*,fmt) c(i, :)
  end do
  write(*,'(A)') '# Difference to input'
  do i = 1, n
    write(*,fmt) c(i, :) - b2(i, :)
  end do
  write(*, *) 'max difference', maxval(real(c-b2, kind=dp))


  ! collect timings
  write(*, *) 
  write(*, *) 'timings:', timings(1:4)
  ! write(*, *) 'speedup LAPACK vs matmul', timing3/timing2
  ! write(*, *) 'speedup GPU vs LAPACK', timing2/timing1
  ! write(*, *) 'speedup GPU vs matmul', timing3/timing1


  ! clean up memory allocation
  deallocate(a, a2, b, b2, c)

  ! destroy cublas adn cusolver handles
  ierr = cublasDestroy(handle_cublas)
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasDestroy'

  ierr = cusolverDnDestroy(handle_cusolver)
  if (ierr/=0) stop 'Error in cusolverDestroy'

end program test
